from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Annotated, Final, Sequence, ClassVar

from pydantic import Field, PlainValidator, AliasChoices, PlainSerializer, ConfigDict
from strenum import StrEnum
from typing_extensions import Self

from ..config.constants import DEFAULT_TOKEN_NAME
from ..ethereum.bin_str import EthBinStrField, EthBinStr
from ..ethereum.hash import EthTxHashField, EthTxHash, EthAddressField, EthZeroAddressField, EthAddress
from ..ethereum.transaction import EthTx
from ..neon.account import NeonAccount, NeonAccountField
from ..neon.neon_program import NeonProgCfg
from ..neon.transaction_model import NeonTxModel
from ..solana.account import SolAccountModel
from ..solana.instruction import SolAccountMeta
from ..solana.pubkey import SolPubKeyField, SolPubKey
from ..solana.transaction import SolTx
from ..utils.cached import cached_property, cached_method
from ..utils.format import bytes_to_hex
from ..utils.pydantic import HexUIntField, BytesField, DecIntField, BaseModel as _BaseModel, DecUIntField

_LOG = logging.getLogger(__name__)


class _BaseRespModel(_BaseModel):
    _model_config = _BaseModel.model_config.copy()
    _model_config.pop("extra")

    model_config = ConfigDict(
        extra="allow",
        **_model_config,
    )


def _gen_unique_id() -> str:
    value = str(uuid.uuid4())
    _LOG.debug("generate ID %s for core-api", value)
    return value


class _BaseRequestModel(_BaseModel):
    ctx_id: str = Field(serialization_alias="id", default_factory=_gen_unique_id)


class CoreApiResultCode(StrEnum):
    Success = "success"
    Unknown = "unknown"

    @classmethod
    def from_raw(cls, value: str | CoreApiResultCode) -> Self:
        if isinstance(value, cls):
            return value

        try:
            value = value.lower()
            return cls(value)
        except (BaseException,):
            _LOG.error("unknown result %s from core api", value.upper())
            return cls.Unknown


CoreApiResultField = Annotated[CoreApiResultCode, PlainValidator(CoreApiResultCode.from_raw)]


class CoreApiResp(_BaseRespModel):
    result: CoreApiResultField
    error: str = None
    error_code: DecIntField | None = None

    value: dict | list[dict] | str | list[int] | None = None
    logs: list[dict] = Field(default_factory=list)


class _AccountModel(_BaseModel):
    address: EthZeroAddressField
    chain_id: DecUIntField

    @classmethod
    def from_raw(cls, raw: _AccountModel | NeonAccount) -> Self:
        if isinstance(raw, _AccountModel):
            return raw
        return cls(address=raw.eth_address, chain_id=raw.chain_id)


class NeonAccountListRequest(_BaseRequestModel):
    account_list: list[_AccountModel] = Field(serialization_alias="account")
    slot: DecUIntField | None

    @classmethod
    def from_raw(cls, account_list: Sequence[NeonAccount], slot: int | None) -> Self:
        return cls(account_list=[_AccountModel.from_raw(a) for a in account_list], slot=slot)


class NeonAccountStatus(StrEnum):
    Ok = "Ok"
    Legacy = "Legacy"
    Empty = "Empty"

    @classmethod
    def from_raw(cls, value: str | NeonAccountStatus) -> Self:
        if isinstance(value, cls):
            return value

        try:
            return cls(value)
        except (BaseException,):
            _LOG.error("unknown neon account status %s from core-api", value.upper())
            return cls.Empty


# Type for Pydantic, it just annotates rules for deserialization
NeonAccountStatusField = Annotated[NeonAccountStatus, PlainValidator(NeonAccountStatus.from_raw)]


class NeonAccountModel(_BaseRespModel):
    account: NeonAccountField
    user_sol_address: SolPubKeyField = Field(SolPubKey.default(), validation_alias="user_pubkey")
    status: NeonAccountStatusField
    state_tx_cnt: DecUIntField = Field(validation_alias=AliasChoices("trx_count", "state_tx_cnt"))
    balance: HexUIntField
    sol_address: SolPubKeyField = Field(validation_alias=AliasChoices("solana_address", "sol_address"))
    contract_sol_address: SolPubKeyField = Field(
        validation_alias=AliasChoices("contract_solana_address", "contract_sol_address")
    )

    @classmethod
    def from_dict(cls, data: [str, Any], *, account: NeonAccount | None = None) -> Self:
        if not account:
            return super().from_dict(data)
        return cls._from_acct(account, data)

    @classmethod
    def new_empty(cls, account: NeonAccount) -> Self:
        return cls(
            account=account,
            status=NeonAccountStatus.Empty,
            sol_address=SolPubKey.default(),
            contract_sol_address=SolPubKey.default(),
            state_tx_cnt=0,
            balance=0,
        )

    @classmethod
    def _from_acct(cls, account: NeonAccount, data: [str, Any]):
        data["account"] = account
        return cls.model_validate(data)

    @property
    def chain_id(self) -> int:
        return self.account.chain_id

    @property
    def address(self) -> EthAddress:
        return self.account.eth_address


class NeonContractRequest(_BaseRequestModel):
    contract: EthZeroAddressField
    slot: DecUIntField | None


class NeonContractModel(_BaseRespModel):
    account: NeonAccountField
    code: EthBinStrField
    sol_address: SolPubKeyField = Field(validation_alias="solana_address")

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, account: NeonAccount | None = None) -> Self:
        if not account:
            return super().from_dict(data)
        return cls._from_acct(account, data)

    @classmethod
    def _from_acct(cls, account: NeonAccount, data: dict[str, Any]) -> Self:
        chain_id: int = data.pop("chain_id", None) or account.chain_id

        # replace with the actual chain-id
        data["account"] = NeonAccountField.from_raw(account, chain_id)
        return cls.model_validate(data)

    @property
    def chain_id(self) -> int:
        return self.account.chain_id

    @property
    def has_code(self) -> bool:
        return not self.code.is_empty


class NeonStorageAtRequest(_BaseRequestModel):
    contract: EthZeroAddressField
    index: HexUIntField
    slot: DecUIntField | None


class OpEarnAccountModel(_BaseModel):
    status: NeonAccountStatusField
    operator_key: SolPubKeyField
    neon_account: NeonAccountField
    token_sol_address: SolPubKeyField
    balance: DecUIntField

    @property
    def chain_id(self) -> int:
        return self.account.chain_id

    @property
    def eth_address(self) -> EthAddress:
        return self.account.eth_address


class BpfLoader2ProgModel(_BaseModel):
    version: int
    exec_address: SolPubKeyField = SolPubKeyField.default()

    @classmethod
    def from_data(cls, data: bytes) -> Self:
        if len(data) != 36:
            return cls(version=0)

        version = int.from_bytes(data[:4], "little")
        if version != 2:
            return cls(version=version)

        return cls(version=version, exec_address=SolPubKeyField.from_bytes(data[4:]))


class BpfLoader2ExecModel(_BaseModel):
    version: int
    deployed_slot: int = 0
    minimum_size: Final[int] = 8

    @classmethod
    def from_data(cls, data: bytes) -> Self:
        if len(data) < cls.minimum_size:
            return cls(version=0)

        version = int.from_bytes(data[:4], "little")
        if version != 3:
            return cls(version=version)

        return cls(version=version, deployed_slot=int.from_bytes(data[4:8], "little"))


class TokenModel(_BaseRespModel):
    chain_id: DecIntField = Field(serialization_alias="id", validation_alias=AliasChoices("id", "chain_id"))
    mint: SolPubKeyField = Field(serialization_alias="token", validation_alias=AliasChoices("token", "mint"))
    name: str
    is_default: bool = Field(default=False, exclude=True)


class EvmConfigModel(_BaseRespModel):
    deployed_slot: DecIntField

    treasury_pool_cnt: DecIntField
    treasury_pool_seed: BytesField
    treasury_payment: DecIntField
    account_seed_version: DecIntField
    evm_step_cnt: DecIntField
    holder_msg_size: DecIntField
    gas_limit_multiplier_wo_chain_id: DecIntField

    evm_param_dict: dict[str, str] = Field(validation_alias=AliasChoices("config", "evm_param_dict"))
    token_list: list[TokenModel] = Field(validation_alias=AliasChoices("chains", "token_list"))

    status: str = Field(default="Ok")
    environment: str = Field(default="Unknown")

    version: str
    revision: str

    _default_chain_id: int = 0
    _default: ClassVar[EvmConfigModel | None] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, deployed_slot: int | None = None) -> Self:
        if deployed_slot is None:
            return super().from_dict(data)
        if "chains" in data:
            return cls._from_core_dict(deployed_slot, data)
        return cls._from_cmd_dict(deployed_slot, data)

    @classmethod
    def _from_core_dict(cls, deployed_slot: int, data: dict[str, Any]) -> Self:
        data["deployed_slot"] = deployed_slot
        config = data["config"]
        cls._option_convertor(config, data)

        return cls.model_validate(data)

    @classmethod
    def _from_cmd_dict(cls, deployed_slot: int, data: dict[str, Any]) -> Self:
        config = data
        opt_dict = dict()
        cls._option_convertor(config, opt_dict)

        def_chain_id = int(config["NEON_CHAIN_ID"])
        base_token_info = dict(
            name=DEFAULT_TOKEN_NAME,
            chain_id=def_chain_id,
            mint=config["NEON_TOKEN_MINT"],
            is_default=True,
        )

        data = dict(
            deployed_slot=deployed_slot,
            evm_param_dict=config,
            token_list=[base_token_info],
        )
        data.update(opt_dict)

        self = cls.model_validate(data)
        object.__setattr__(self, "_default_chain_id", def_chain_id)
        return self

    @classmethod
    def default(cls) -> Self:
        if not cls._default:
            opt_dict = dict()
            cls._option_convertor(dict(), opt_dict)

            data = dict(
                deployed_slot=-1,
                evm_param_dict=dict(),
                token_list=list(),
            )
            data.update(opt_dict)

            cls._default = cls.model_validate(data)
        return cls._default

    @property
    def is_empty(self) -> bool:
        return self.deployed_slot == -1

    @cached_property
    def package_version(self) -> str:
        return "Neon-EVM/v" + self.version + "-" + self.revision

    @cached_property
    def token_dict(self) -> dict[str, TokenModel]:
        return {token.name: token for token in self._normalized_token_list}

    @cached_property
    def chain_dict(self) -> dict[int, TokenModel]:
        return {token.chain_id: token for token in self._normalized_token_list}

    @cached_property
    def _normalized_token_list(self) -> Sequence[TokenModel]:
        if self._default_chain_id:
            return tuple(self.token_list)

        def_chain_id = 0
        token_list: list[TokenModel] = list()
        for token in self.token_list:
            name = token.name.upper()
            if is_default := (name == DEFAULT_TOKEN_NAME):
                def_chain_id = token.chain_id
            token = TokenModel(
                chain_id=token.chain_id,
                mint=token.mint,
                name=name,
                is_default=is_default,
            )
            token_list.append(token)

        assert def_chain_id, "DEFAULT TOKEN NOT FOUND!"
        object.__setattr__(self, "_default_chain_id", def_chain_id)

        return tuple(token_list)

    @cached_property
    def default_chain_id(self) -> int:
        if not self._default_chain_id:
            _ = self._normalized_token_list
        return self._default_chain_id

    @property
    def default_token_name(self) -> str:
        return DEFAULT_TOKEN_NAME

    @cached_property
    def neon_prog_cfg(self) -> NeonProgCfg:
        return NeonProgCfg(
            treasury_pool_cnt=self.treasury_pool_cnt,
            treasury_pool_seed=self.treasury_pool_seed,
            treasury_payment=self.treasury_payment,
            evm_version=self.version,
        )

    @classmethod
    def _option_convertor(cls, src_dict: dict[str, Any], dst_dict: dict[str, Any]) -> None:
        key_list = (
            ("NEON_PKG_VERSION", "version", "0.0.0-unknown"),
            ("NEON_REVISION", "revision", "unknown"),
            ("NEON_TREASURY_POOL_COUNT", "treasury_pool_cnt", -1),
            ("NEON_TREASURY_POOL_SEED", "treasury_pool_seed", bytes()),
            ("NEON_PAYMENT_TO_TREASURE", "treasury_payment", -1),
            ("NEON_EVM_STEPS_MIN", "evm_step_cnt", -1),
            ("NEON_HOLDER_MSG_SIZE", "holder_msg_size", -1),
            ("NEON_ACCOUNT_SEED_VERSION", "account_seed_version", -1),
            ("NEON_GAS_LIMIT_MULTIPLIER_NO_CHAINID", "gas_limit_multiplier_wo_chain_id", -1),
        )

        for src_key, dst_key, default in key_list:
            if dst_key not in dst_dict:
                dst_dict[dst_key] = src_dict.get(src_key, default)


class HolderAccountRequest(_BaseRequestModel):
    pubkey: SolPubKeyField

    @classmethod
    def from_raw(cls, pubkey: SolPubKey) -> Self:
        return cls(pubkey=pubkey)


class HolderAccountStatus(StrEnum):
    Empty = "Empty"
    Error = "Error"
    Holder = "Holder"
    Active = "Active"
    Finalized = "Finalized"

    @classmethod
    def from_raw(cls, value: str | HolderAccountStatus) -> Self:
        if isinstance(value, cls):
            return value

        try:
            return cls(value)
        except (BaseException,):
            _LOG.error("unknown holder status %s from core-api", value)
            return cls.Error


# Type for Pydantic, it just annotates rules for deserialization
HolderAccountStatusField = Annotated[HolderAccountStatus, PlainValidator(HolderAccountStatus.from_raw)]


class CoreApiHexStr(EthBinStr):
    @cached_method
    def _to_string(self) -> str:
        return bytes_to_hex(self._data, prefix="")

    def to_string(self, default: str | None = "") -> str | None:
        return super().to_string(default=default)


CoreApiHexStrField = Annotated[
    CoreApiHexStr,
    PlainValidator(CoreApiHexStr.from_raw),
    PlainSerializer(lambda v: v.to_string() or None),
]


class CoreApiTxModel(_BaseRespModel):
    from_address: EthZeroAddressField = Field(
        validation_alias=AliasChoices("from", "from_address"),
        serialization_alias="from",
    )
    nonce: DecUIntField | None
    to_address: EthAddressField = Field(
        default=EthAddress.default(),
        validation_alias=AliasChoices("to", "to_address"),
        serialization_alias="to",
    )
    value: HexUIntField
    data: CoreApiHexStrField
    gas_limit: HexUIntField | None
    gas_price: HexUIntField | None
    max_fee_per_gas: HexUIntField = Field(default=0)
    max_priority_fee_per_gas: HexUIntField = Field(default=0)

    chain_id: DecUIntField | None = None

    @classmethod
    def from_neon_tx(cls, tx: NeonTxModel, chain_id: int | None) -> Self:
        return cls(
            from_address=tx.from_address,
            nonce=tx.nonce,
            to_address=tx.to_address,
            value=tx.value,
            data=tx.call_data.to_bytes(),
            gas_limit=tx.gas_limit,
            gas_price=tx.gas_price or 0,
            max_fee_per_gas=tx.max_fee_per_gas or 0,
            max_priority_fee_per_gas=tx.max_priority_fee_per_gas or 0,
            chain_id=chain_id,
        )

    @cached_property
    def cost(self) -> int:
        return EthTx.calc_cost(self)

    @cached_property
    def has_priority_fee(self) -> bool:
        return EthTx.has_priority_fee(self)

    @cached_property
    def base_fee_per_gas(self) -> int:
        return EthTx.calc_base_fee_per_gas(self)


class CoreApiBlockModel(_BaseModel):
    timestamp: DecUIntField | None = Field(default=None, serialization_alias="time")
    slot: DecUIntField | None = Field(default=None, serialization_alias="number")

    _default: ClassVar[CoreApiBlockModel | None] = None

    @classmethod
    def default(cls) -> Self:
        if not cls._default:
            cls._default = CoreApiBlockModel()
        return cls._default

    @classmethod
    def from_raw(cls, raw_list: list[str] | None) -> Self:
        if not raw_list:
            return cls.default()

        return cls(
            timestamp=int(raw_list[0], 16),
            slot=int(raw_list[1], 16),
        )

    @property
    def is_empty(self) -> bool:
        return self.slot is None


class HolderAccountModel(_BaseRespModel):
    address: SolPubKeyField

    status: HolderAccountStatusField
    size: DecUIntField = Field(default=0, validation_alias="len")
    owner: SolPubKeyField = Field(default=SolPubKey.default())

    neon_tx_hash: EthTxHashField = Field(default=EthTxHash.default(), validation_alias="tx")
    tx_type: DecUIntField = Field(default=0)
    tx: CoreApiTxModel | None = Field(default=None, validation_alias="tx_data")
    block: CoreApiBlockModel

    chain_id: DecUIntField | None = Field(default=0)
    evm_step_cnt: DecUIntField = Field(default=0, validation_alias="steps_executed")
    account_key_list: list[SolPubKeyField] = Field(default_factory=list, validation_alias="accounts")

    @classmethod
    def new_empty(cls, address: SolPubKey) -> Self:
        return cls(
            address=address,
            status=HolderAccountStatus.Empty,
            owner=SolPubKey.default(),
            block=CoreApiBlockModel.default(),
        )

    @classmethod
    def from_dict(cls, address: SolPubKey, def_chain_id: int, data: dict) -> Self:  # noqa
        data["address"] = address
        data["block"] = CoreApiBlockModel.from_raw(data.pop("block_params", None))
        data["chain_id"] = data.get("chain_id", def_chain_id)
        return cls.model_validate(data)

    @cached_property
    def sender(self) -> NeonAccount:
        if self.tx is None:
            return NeonAccount.default()

        return NeonAccount.from_raw(self.tx.from_address, self.chain_id)

    @cached_property
    def receiver(self) -> NeonAccount:
        if self.tx is None:
            return NeonAccount.default()

        elif not self.tx.to_address.is_empty:
            return NeonAccount.from_raw(self.tx.to_address, self.chain_id)

        contract_addr = EthTx.calc_contract_address(self.tx)
        return NeonAccount.from_raw(contract_addr, self.chain_id)

    @property
    def is_empty(self) -> bool:
        return self.status in (HolderAccountStatus.Empty, HolderAccountStatus.Error)

    @property
    def is_active(self) -> bool:
        return self.status == HolderAccountStatus.Active

    @property
    def is_finalized(self) -> bool:
        return self.status == HolderAccountStatus.Finalized


class _CrateModel(_BaseRespModel):
    version: str


class _VersionModel(_BaseRespModel):
    commit_id: str


class CoreApiBuildModel(_BaseRespModel):
    crate_info: _CrateModel
    version_control: _VersionModel


class EmulSolAccountModel(_BaseModel):
    balance: DecUIntField = Field(serialization_alias="lamports")
    data: CoreApiHexStrField
    owner: SolPubKeyField
    executable: bool
    rent_epoch: DecUIntField

    @classmethod
    def from_raw(cls, raw: SolAccountModel | None) -> Self | None:
        if raw is None:
            return None

        return cls(
            balance=raw.balance,
            data=raw.data,
            owner=raw.owner,
            executable=raw.executable,
            rent_epoch=raw.rent_epoch,
        )


class EmulNeonAccountModel(_BaseModel):
    nonce: DecUIntField | None = None
    balance: HexUIntField | None = None


class EmulTraceCfgModel(_BaseModel):
    neon_account_dict: dict[EthZeroAddressField, EmulNeonAccountModel] = Field(serialization_alias="stateOverrides")
    block: CoreApiBlockModel | None = Field(default=None, serialization_alias="blockOverrides")


class EmulNeonCallRequest(_BaseRequestModel):
    tx: CoreApiTxModel
    evm_step_limit: DecUIntField = Field(serialization_alias="step_limit")
    token_list: list[TokenModel] = Field(serialization_alias="chains")
    trace_cfg: EmulTraceCfgModel | None = Field(serialization_alias="trace_config")
    preload_sol_address_list: list[SolPubKeyField] = Field(serialization_alias="accounts")
    sol_account_dict: dict[SolPubKeyField, EmulSolAccountModel | None] | None = Field(
        serialization_alias="solana_overrides"
    )
    slot: DecUIntField | None
    provide_account_info: str | None = None


class EmulNeonCallExitCode(StrEnum):
    Revert = "revert"
    Succeed = "succeed"
    StepLimitExceeded = "step limit exceeded"
    Unknown = "unknown"

    @classmethod
    def from_raw(cls, value: str | EmulNeonCallExitCode) -> Self:
        if isinstance(value, cls):
            return value

        try:
            value = value.lower()
            return cls(value)
        except (BaseException,):
            _LOG.error(f"unknown emulator exit {value}")
            return cls.Unknown


EmulNeonCallExitCodeField = Annotated[EmulNeonCallExitCode, PlainValidator(EmulNeonCallExitCode.from_raw)]


class EmulAccountMetaModel(_BaseRespModel):
    pubkey: SolPubKeyField
    is_writable: bool
    is_legacy: bool

    def to_sol_account_meta(self) -> SolAccountMeta:
        return SolAccountMeta(pubkey=self.pubkey, is_writable=self.is_writable, is_signer=False)


class EmulNeonCallResp(_BaseRespModel):
    exit_code: EmulNeonCallExitCodeField = Field(validation_alias="exit_status")
    external_sol_call: bool = Field(validation_alias="external_solana_call")
    revert_before_sol_call: bool = Field(validation_alias="reverts_before_solana_calls")
    revert_after_sol_call: bool = Field(validation_alias="reverts_after_solana_calls")
    is_block_used: bool = Field(False, validation_alias="is_timestamp_number_used")

    result: EthBinStrField
    evm_step_cnt: DecUIntField = Field(validation_alias="steps_executed")
    used_gas: DecUIntField
    iter_cnt: DecUIntField = Field(alias="iterations")

    raw_meta_list: list[EmulAccountMetaModel] = Field(validation_alias="solana_accounts", default_factory=list)

    @cached_property
    def sol_account_meta_list(self) -> Sequence[SolAccountMeta]:
        return tuple([a.to_sol_account_meta() for a in self.raw_meta_list])


class EmulSolTxListRequest(_BaseRespModel):
    cu_limit: DecUIntField = Field(serialization_alias="compute_units")
    account_cnt_limit: DecUIntField = Field(serialization_alias="account_limit")
    verify: bool
    blockhash: CoreApiHexStrField
    tx_list: list[CoreApiHexStrField] = Field(serialization_alias="transactions")


class EmulSolTxMetaModel(_BaseRespModel):
    error: dict | None
    log_list: list[str] = Field(default_factory=list, validation_alias="logs")
    used_cu_limit: DecUIntField = Field(validation_alias="executed_units")


class EmulSolTxListResp(_BaseRespModel):
    meta_list: list[EmulSolTxMetaModel] = Field(validation_alias="transactions")


@dataclass(frozen=True)
class EmulSolTxInfo:
    tx: SolTx
    meta: EmulSolTxMetaModel
