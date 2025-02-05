from typing import ClassVar

from common.neon.neon_program import NeonEvmIxCode
from common.solana.instruction import SolTxIx
from .strategy_base import SolTxCfg
from .strategy_simple import SimpleTxStrategy
from .strategy_stage_alt import alt_strategy
from .strategy_stage_write_holder import WriteHolderTxPrepStage


class SimpleHolderTxStrategy(SimpleTxStrategy):
    name: ClassVar[str] = NeonEvmIxCode.TxExecFromAccount.name

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prep_stage_list.append(WriteHolderTxPrepStage(*args, **kwargs))

    def _build_tx_ix(self, tx_cfg: SolTxCfg) -> SolTxIx:
        return self._ctx.neon_prog.make_tx_exec_from_account_ix()

    async def _validate(self) -> bool:
        return (
            self._validate_not_stuck_tx()
            and self._validate_no_sol_call()
            and self._validate_has_chain_id()
            and self._validate_no_resize_iter()
        )


@alt_strategy
class AltSimpleHolderTxStrategy(SimpleHolderTxStrategy):
    pass
