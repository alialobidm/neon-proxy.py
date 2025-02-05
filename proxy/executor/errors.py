from common.ethereum.hash import EthTxHash
from common.neon_rpc.api import HolderAccountModel
from common.solana.pubkey import SolPubKey
from common.utils.cached import cached_method


class WrongStrategyError(Exception):
    def __str__(self) -> str:
        return "Wrong strategy"


class BadResourceError(Exception):
    def __str__(self) -> str:
        return "Bad resource"


class StuckTxError(Exception):
    def __init__(self, holder: HolderAccountModel) -> None:
        super().__init__()
        self._neon_tx_hash = holder.neon_tx_hash
        self._chain_id = holder.chain_id
        self._address = holder.address

    @property
    def neon_tx_hash(self) -> EthTxHash:
        return self._neon_tx_hash

    @property
    def chain_id(self) -> int:
        return self._chain_id

    @property
    def address(self) -> SolPubKey:
        return self._address

    @cached_method
    def to_string(self) -> str:
        return f"Holder {self._address} contains stuck tx {self._neon_tx_hash}"

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()
