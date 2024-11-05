from __future__ import annotations

import logging
from typing import Final, ClassVar

from common.neon.neon_program import NeonEvmIxCode, NeonProg
from common.solana.transaction_legacy import SolLegacyTx
from common.solana_rpc.errors import SolCbExceededError
from common.solana_rpc.transaction_list_sender import SolTxSendState
from .errors import WrongStrategyError
from .strategy_base import BaseTxStrategy, SolTxCfg
from .strategy_stage_alt import alt_strategy
from .strategy_stage_new_account import NewAccountTxPrepStage
from ..base.ex_api import ExecTxRespCode

_LOG = logging.getLogger(__name__)


class SimpleTxStrategy(BaseTxStrategy):
    name: ClassVar[str] = NeonEvmIxCode.TxExecFromData.name

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prep_stage_list.append(NewAccountTxPrepStage(*args, **kwargs))

    async def execute(self) -> ExecTxRespCode:
        assert self.is_valid

        if not await self._recheck_tx_list(self.name):
            await self._emulate_and_send_tx_list()

        tx_send_state_list = self._sol_tx_list_sender.tx_state_list
        status = SolTxSendState.Status

        for tx_state in tx_send_state_list:
            if tx_state.status == status.GoodReceipt:
                if not (sol_neon_ix := self._find_sol_neon_ix(tx_state)):
                    _LOG.warning("no!? NeonTx instruction in %s", tx_state.tx)
                    return ExecTxRespCode.Failed
                elif not sol_neon_ix.neon_tx_return.is_empty:
                    _LOG.debug("found NeonTx-Return in %s", tx_state.tx)
                    return ExecTxRespCode.Done
                else:
                    _LOG.warning("truncated!? NeonTx-Return in %s", tx_state.tx)
                    return ExecTxRespCode.Failed

        _LOG.debug("failed!? NeonTx-Return, try next strategy...")
        raise WrongStrategyError()

    async def cancel(self) -> None:
        _LOG.debug("canceling of a simple NeonTx...")
        return None

    async def _emulate_and_send_tx_list(self) -> bool:
        base_cfg = self._init_sol_tx_cfg(cu_limit=self._ctx.cfg.cu_limit)
        base_tx = self._build_tx(base_cfg)

        emul_tx = await self._emulate_tx_list(base_tx)
        used_cu_limit: Final[int] = emul_tx.meta.used_cu_limit

        evm_step_cnt: Final[int] = self._ctx.total_evm_step_cnt
        max_cu_limit: Final[int] = base_cfg.cu_limit
        # let's decrease the available cu-limit on 5% percents, because Solana decrease it
        threshold_cu_limit: Final[int] = int(max_cu_limit * 0.95)

        if used_cu_limit > threshold_cu_limit:
            _LOG.debug(
                "simple: %d EVM steps, %d CUs is bigger than the upper limit %d",
                evm_step_cnt,
                used_cu_limit,
                threshold_cu_limit,
            )
            raise SolCbExceededError()

        round_coeff: Final[int] = 10_000
        inc_coeff: Final[int] = 100_000
        round_cu_limit = min((used_cu_limit // round_coeff) * round_coeff + inc_coeff, max_cu_limit)
        _LOG.debug("simple: %d EVM steps, %d CUs (round to %s CUs)", evm_step_cnt, used_cu_limit, round_cu_limit)

        gas_limit = self._find_gas_limit(emul_tx)

        optimal_cfg = await self._update_cu_price(base_cfg, cu_limit=round_cu_limit, gas_limit=gas_limit)
        tx = self._build_tx(optimal_cfg)
        return await self._send_tx_list(tx)

    async def _update_cu_price(self, tx_cfg: SolTxCfg, *, cu_limit: int, gas_limit: int) -> SolTxCfg:
        cu_price: int = await self._calc_cu_price(cu_limit=cu_limit, gas_limit=gas_limit)
        return tx_cfg.update(cu_limit=cu_limit, gas_limit=gas_limit, cu_price=cu_price)

    def _build_tx(self, tx_cfg: SolTxCfg) -> SolLegacyTx:
        return self._build_cu_tx(self._ctx.neon_prog.make_tx_exec_from_data_ix(), tx_cfg)

    async def _validate(self) -> bool:
        return (
            self._validate_not_stuck_tx()
            and self._validate_no_sol_call()
            and self._validate_has_chain_id()
            and self._validate_no_resize_iter()
            and self._validate_neon_tx_size()
        )


@alt_strategy
class AltSimpleTxStrategy(SimpleTxStrategy):
    pass
