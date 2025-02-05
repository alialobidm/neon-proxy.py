from __future__ import annotations

import abc
import dataclasses
import logging
from typing import Sequence, Final, ClassVar

from typing_extensions import Self

from common.neon.evm_log_decoder import NeonEvmLogDecoder
from common.neon.neon_program import NeonIxMode, NeonProg
from common.neon.transaction_decoder import SolNeonTxMetaInfo, SolNeonTxIxMetaInfo
from common.neon_rpc.api import EmulSolTxInfo
from common.solana.cb_program import SolCbProg
from common.solana.commit_level import SolCommit
from common.solana.pubkey import SolPubKey
from common.solana.signer import SolSigner
from common.solana.transaction import SolTx, SolTxIx
from common.solana.transaction_decoder import SolTxMetaInfo, SolTxIxMetaInfo
from common.solana.transaction_legacy import SolLegacyTx
from common.solana.transaction_meta import SolRpcTxSlotInfo
from common.solana_rpc.errors import SolCbExceededError
from common.solana_rpc.transaction_list_sender import SolTxSendState, SolTxListSender
from common.utils.cached import cached_property
from .server_abc import ExecutorComponent, ExecutorServerAbc
from .transaction_executor_ctx import NeonExecTxCtx
from ..base.ex_api import ExecTxRespCode

_LOG = logging.getLogger(__name__)


class BaseTxPrepStage(ExecutorComponent, abc.ABC):
    def __init__(self, server: ExecutorServerAbc, ctx: NeonExecTxCtx):
        super().__init__(server)
        self._ctx = ctx

    @property
    def _cu_price(self) -> int:
        return self._ctx.token.simple_cu_price

    @abc.abstractmethod
    def get_tx_name_list(self) -> Sequence[str]:
        pass

    @abc.abstractmethod
    async def build_tx_list(self) -> list[list[SolTx]]:
        pass

    @abc.abstractmethod
    async def update_after_emulation(self) -> bool:
        pass


@dataclasses.dataclass(frozen=True)
class SolTxCfg:
    name: str
    ix_mode: NeonIxMode

    cu_limit: int
    cu_price: int
    heap_size: int

    gas_limit: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)  # noqa

    def update(self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)


class BaseTxStrategy(ExecutorComponent, abc.ABC):
    name: ClassVar[str] = "UNKNOWN STRATEGY"
    is_simple: ClassVar[bool] = True

    def __init__(self, server: ExecutorServerAbc, ctx: NeonExecTxCtx) -> None:
        super().__init__(server)
        self._ctx = ctx
        self._validation_error_msg: str | None = None
        self._prep_stage_list: list[BaseTxPrepStage] = list()

    @property
    def validation_error_msg(self) -> str:
        assert not self.is_valid
        return self._validation_error_msg

    @property
    def is_valid(self) -> bool:
        return self._validation_error_msg is None

    async def validate(self) -> bool:
        self._validation_error_msg = None
        try:
            if result := await self._validate():
                result = self._validate_tx_size()
            assert result == (self._validation_error_msg is None)

            return result
        except BaseException as e:
            self._validation_error_msg = str(e)
            return False

    async def prep_before_emulation(self) -> None:
        assert self.is_valid

        # recheck already sent transactions
        tx_name_list: list[str] = list()
        for stage in self._prep_stage_list:
            tx_name_list.extend(stage.get_tx_name_list())
        await self._recheck_tx_list(tuple(tx_name_list))

        # generate new transactions
        tx_list_list = await self._build_prep_tx_list()

        for tx_list in tx_list_list:
            await self._send_tx_list(tx_list)

    async def update_after_emulation(self) -> bool:
        assert self.is_valid

        result = True
        for stage in self._prep_stage_list:
            result = await stage.update_after_emulation() and result
        return result

    @abc.abstractmethod
    async def execute(self) -> ExecTxRespCode:
        pass

    @abc.abstractmethod
    async def cancel(self) -> ExecTxRespCode | None:
        pass

    @cached_property
    def _sol_tx_list_sender(self) -> SolTxListSender:
        return SolTxListSender(
            self._cfg,
            self._stat_client,
            self._ctx.sol_watch_session,
            self._ctx.sol_tx_list_signer,
        )

    def _validate_tx_size(self) -> bool:
        with self._ctx.test_mode():
            base_cfg = self._init_sol_tx_cfg()
            ix = self._build_tx_ix(base_cfg)
            tx = self._build_cu_tx(ix, base_cfg)
            tx.validate(SolSigner.fake())  # <- there can be SolTxSizeError
        return True

    def _validate_has_chain_id(self) -> bool:
        if self._ctx.has_chain_id:
            return True

        self._validation_error_msg = "Transaction without chain-id"
        return False

    def _validate_not_stuck_tx(self) -> bool:
        if not self._ctx.is_stuck_tx:
            return True

        self._validation_error_msg = "Stuck transaction"
        return False

    def _validate_no_sol_call(self) -> bool:
        if not self._ctx.has_external_sol_call:
            return True
        self._validation_error_msg = "Has external Solana call"
        return False

    def _validate_gas_price(self) -> bool:
        if self._ctx.holder_tx.base_fee_per_gas:
            return True
        self._validation_error_msg = "Fee less transaction"
        return False

    def _validate_has_sol_call(self) -> bool:
        if self._ctx.has_external_sol_call:
            return True
        self._validation_error_msg = "Doesn't have external Solana call"
        return False

    def _validate_no_resize_iter(self) -> bool:
        if self._ctx.resize_iter_cnt <= 0:
            return True
        self._validation_error_msg = f"Has {self._ctx.resize_iter_cnt} resize iterations"
        return False

    def _validate_neon_tx_size(self) -> bool:
        neon_tx_size = len(self._ctx.neon_prog.holder_msg)
        if len(self._ctx.neon_prog.holder_msg) < self._base_sol_pkt_size:
            return True
        self._validation_error_msg = f"NeonTx has size {neon_tx_size} > {self._base_sol_pkt_size}"
        return False

    @cached_property
    def _base_sol_pkt_size(self) -> int:
        return SolTx.PktSize - NeonProg.BaseAccountCnt * SolPubKey.KeySize

    async def _build_prep_tx_list(self) -> list[list[SolTx]]:
        tx_list_list: list[list[SolTx]] = list()

        for stage in self._prep_stage_list:
            new_tx_list_list = await stage.build_tx_list()

            while len(new_tx_list_list) > len(tx_list_list):
                tx_list_list.append(list())
            for tx_list, new_tx_list in zip(tx_list_list, new_tx_list_list):
                tx_list.extend(new_tx_list)

        return tx_list_list

    async def _recheck_tx_list(self, tx_name_list: Sequence[str] | str) -> bool:
        tx_list_sender = self._sol_tx_list_sender
        tx_list_sender.clear()

        if not isinstance(tx_name_list, Sequence):
            tx_name_list = tuple([tx_name_list])

        if not (tx_list := self._ctx.pop_sol_tx_list(tx_name_list)):
            return False

        try:
            return await tx_list_sender.recheck(tx_list)
        finally:
            self._store_sol_tx_list()

    async def _send_tx_list(self, tx_list: Sequence[SolTx] | SolTx) -> bool:
        tx_list_sender = self._sol_tx_list_sender
        tx_list_sender.clear()

        if not isinstance(tx_list, Sequence):
            tx_list = tuple([tx_list])

        try:
            return await tx_list_sender.send(tx_list)
        finally:
            self._store_sol_tx_list()

    def _store_sol_tx_list(self):
        tx_list_sender = self._sol_tx_list_sender
        self._ctx.add_sol_tx_list(
            [
                (tx_state.tx, tx_state.status == tx_state.status.GoodReceipt)
                for tx_state in tx_list_sender.tx_state_list
                # we shouldn't retry txs with the exceed Compute Budget error
                if tx_state.status != tx_state.status.CbExceededError
            ]
        )

    # async def _estimate_cu_price(self) -> int:
    #     # We estimate the cu_price from the recent blocks.
    #     # Solana currently does not really take into account writeable account list,
    #     # so the decent estimation level should be achieved by taking a weighted average from
    #     # the percentiles of compute unit prices across recent blocks.
    #     est_block_cnt = self._ctx.cfg.cu_price_estimator_block_cnt
    #     est_percentile = self._ctx.cfg.cu_price_estimator_percentile
    #     block_list = await self._ctx.db.get_block_cu_price_list(est_block_cnt)
    #
    #     return int(
    #         CuPricePercentileModel.get_weighted_percentile(
    #             est_percentile, len(block_list), map(lambda v: v.cu_price_list, block_list)
    #         )
    #     )

    def _init_sol_tx_cfg(
        self,
        *,
        name: str = "",
        ix_mode: NeonIxMode = NeonIxMode.Default,
        cu_limit: int = SolCbProg.MaxCuLimit,
        cu_price: int = SolCbProg.BaseCuPrice,
        heap_size: int = SolCbProg.MaxHeapSize,
        gas_limit: int = NeonProg.BaseGas,
    ) -> SolTxCfg:
        return SolTxCfg(
            name=name or self.name,
            ix_mode=ix_mode,
            cu_limit=cu_limit,
            cu_price=cu_price,
            heap_size=heap_size,
            gas_limit=gas_limit,
        )

    async def _calc_cu_price(self, cu_limit: int, gas_limit: int) -> int:
        token = self._ctx.token

        # calculate a required cu-price from the Solana statistics
        req_cu_price = await self._cu_price_client.get_cu_price(self._ctx.rw_account_key_list)

        tx = self._ctx.holder_tx
        assert tx.base_fee_per_gas >= 0

        if tx.has_priority_fee:
            priority_fee = tx.max_priority_fee_per_gas * 100 / tx.base_fee_per_gas
            gas_limit = NeonProg.SignatureGas
            _LOG.debug(
                "use %s%% priority-fee for priority gas-price %d",
                priority_fee,
                tx.max_priority_fee_per_gas,
            )
        else:
            # calculate a transaction cu-price based on the tx gas-price
            priority_fee = max(tx.base_fee_per_gas - token.profitable_gas_price, 0) / token.pct_gas_price
            _LOG.debug("use %s%% priority-fee for legacy gas-price %d", priority_fee, tx.base_fee_per_gas)

        if priority_fee > 0.0:
            # see gas-price-calculator for details
            tx_cu_price = int(priority_fee * gas_limit * SolCbProg.MicroLamport / cu_limit / 100)
        else:
            tx_cu_price = 0

        # cu_price should be more than 0, otherwise the Compute Budget instructions are skipped
        # and neon-evm does not digest it.
        cu_price = max(min(req_cu_price, tx_cu_price), 1)

        _LOG.debug(
            "use %s CU-price for %s CU-limit, %s Gas-limit, %s accounts",
            cu_price,
            cu_limit,
            gas_limit,
            len(self._ctx.rw_account_key_list),
        )
        return cu_price

    @staticmethod
    def _build_cu_tx(ix: SolTxIx, tx_cfg: SolTxCfg) -> SolLegacyTx:
        ix_list: list[SolTxIx] = list()

        if tx_cfg.cu_price:
            ix_list.append(SolCbProg.make_cu_price_ix(tx_cfg.cu_price))
        if tx_cfg.cu_limit:
            ix_list.append(SolCbProg.make_cu_limit_ix(tx_cfg.cu_limit))
        if tx_cfg.heap_size:
            ix_list.append(SolCbProg.make_heap_size_ix(tx_cfg.heap_size))

        ix_list.append(ix)

        return SolLegacyTx(name=tx_cfg.name, ix_list=ix_list)

    async def _emulate_tx_list(
        self, tx_list: Sequence[SolTx] | SolTx, *, mult_factor: int = 0
    ) -> Sequence[EmulSolTxInfo] | EmulSolTxInfo:
        if not isinstance(tx_list, Sequence):
            is_single_tx: Final[bool] = True
            tx_list = tuple([tx_list])
        else:
            is_single_tx: Final[bool] = False

        blockhash, _ = await self._sol_client.get_recent_blockhash(SolCommit.Finalized)
        for tx in tx_list:
            tx.set_recent_blockhash(blockhash)
        tx_list = await self._ctx.sol_tx_list_signer.sign_tx_list(tx_list)

        acct_cnt_limit: Final[int] = 255  # not critical here, it's already tested on the validation step
        cu_limit = SolCbProg.MaxCuLimit * (mult_factor or len(tx_list))

        try:
            emul_tx_list = await self._core_api_client.emulate_sol_tx_list(cu_limit, acct_cnt_limit, blockhash, tx_list)
            return emul_tx_list[0] if is_single_tx else emul_tx_list
        except BaseException as exc:
            _LOG.warning("error on emulate solana tx list", exc_info=exc)
            raise SolCbExceededError()

    @staticmethod
    def _find_gas_limit(emul_tx: EmulSolTxInfo) -> int:
        fake_tx_ix = SolTxIxMetaInfo.default()

        try:
            log = NeonEvmLogDecoder().decode(fake_tx_ix, emul_tx.meta.log_list)
        except (BaseException,):
            return NeonProg.BaseGas

        if log.tx_ix_gas.is_empty:
            gas_limit = NeonProg.BaseGas
            _LOG.debug("no GAS information, use default %s", gas_limit)
        else:
            gas_limit = log.tx_ix_gas.gas_used
            _LOG.debug("found GAS %s", gas_limit)
        return gas_limit

    async def _emulate_and_send_single_tx(self, hdr: str, ix: SolTxIx, base_cfg: SolTxCfg) -> bool:
        base_tx = self._build_cu_tx(ix, base_cfg)
        emul_tx = await self._emulate_tx_list(base_tx)
        used_cu_limit: Final[int] = emul_tx.meta.used_cu_limit

        max_cu_limit: Final[int] = base_cfg.cu_limit
        # let's decrease the available cu-limit on 5% percents, because Solana uses it for ComputeBudget calls
        threshold_cu_limit: Final[int] = int(max_cu_limit * 0.95)

        if used_cu_limit > threshold_cu_limit:
            _LOG.debug(
                "%s: %d CUs is bigger than the upper limit %d",
                hdr,
                used_cu_limit,
                threshold_cu_limit,
            )
            raise SolCbExceededError()

        round_coeff: Final[int] = 10_000
        inc_coeff: Final[int] = 100_000
        round_cu_limit = min((used_cu_limit // round_coeff) * round_coeff + inc_coeff, max_cu_limit)
        _LOG.debug("%s: %d CUs (round to %d CUs)", hdr, used_cu_limit, round_cu_limit)

        gas_limit = self._find_gas_limit(emul_tx)

        for cu_limit in (round_cu_limit, max_cu_limit):
            cu_price = await self._calc_cu_price(cu_limit=cu_limit, gas_limit=gas_limit)
            optimal_cfg = base_cfg.update(cu_limit=cu_limit, gas_limit=gas_limit, cu_price=cu_price)

            optimal_tx = self._build_cu_tx(ix, optimal_cfg)
            try:
                return await self._send_tx_list(optimal_tx)
            except SolCbExceededError:
                if cu_limit == max_cu_limit:
                    raise
                _LOG.debug("%s: try the maximum %d CUs", max_cu_limit)

    @staticmethod
    def _find_sol_neon_ix(tx_send_state: SolTxSendState) -> SolNeonTxIxMetaInfo | None:
        if not isinstance(tx_send_state.receipt, SolRpcTxSlotInfo):
            return None

        sol_tx = SolTxMetaInfo.from_raw(tx_send_state.slot, tx_send_state.receipt.transaction)
        sol_neon_tx = SolNeonTxMetaInfo.from_raw(sol_tx)
        return next(iter(sol_neon_tx.sol_neon_ix_list()), None)

    @abc.abstractmethod
    def _build_tx_ix(self, tx_cfg: SolTxCfg) -> SolTxIx:
        pass

    @abc.abstractmethod
    async def _validate(self) -> bool:
        pass
