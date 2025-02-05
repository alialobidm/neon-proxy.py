from typing import ClassVar

from common.neon.neon_program import NeonEvmIxCode
from common.solana.instruction import SolTxIx
from .strategy_base import SolTxCfg
from .strategy_simple_solana_call import SimpleTxSolanaCallStrategy
from .strategy_stage_alt import alt_strategy
from .strategy_stage_write_holder import WriteHolderTxPrepStage


class SimpleHolderTxSolanaCallStrategy(SimpleTxSolanaCallStrategy):
    name: ClassVar[str] = NeonEvmIxCode.TxExecFromAccountSolanaCall.name

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prep_stage_list.append(WriteHolderTxPrepStage(*args, **kwargs))

    def _build_tx_ix(self, tx_cfg: SolTxCfg) -> SolTxIx:
        return self._ctx.neon_prog.make_tx_exec_from_account_solana_call_ix()

    async def _validate(self) -> bool:
        return (
            self._validate_not_stuck_tx()
            and self._validate_gas_price()
            and self._validate_has_chain_id()
            and self._validate_has_sol_call()
            and self._validate_no_resize_iter()
        )


@alt_strategy
class AltSimpleHolderTxSolanaCallStrategy(SimpleHolderTxSolanaCallStrategy):
    pass
