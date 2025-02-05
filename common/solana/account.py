from __future__ import annotations

from typing import ClassVar, Union

import solders.account as _acct
from pydantic import Field, AliasChoices
from typing_extensions import Self

from .pubkey import SolPubKey, SolPubKeyField
from ..utils.pydantic import Base64Field, BaseModel

SolRpcAccountInfo = _acct.Account


class SolAccountModel(BaseModel):
    address: SolPubKeyField = Field(default=SolPubKey.default())
    balance: int = Field(0, validation_alias=AliasChoices("lamports", "balance"))
    data: Base64Field
    owner: SolPubKeyField
    executable: bool = Field(default=False)
    rent_epoch: int = Field(default=0, validation_alias=AliasChoices("rent_epoch", "rentEpoch"))

    _default: ClassVar[SolAccountModel | None] = None

    @classmethod
    def default(cls) -> Self:
        if not cls._default:
            cls._default = cls(
                address=SolPubKey.default(),
                data=bytes(),
                owner=SolPubKey.default(),
            )
        return cls._default

    @classmethod
    def new_empty(cls, address: SolPubKeyField) -> SolAccountModel:
        return cls(
            address=address,
            balance=0,
            data=bytes(),
            owner=SolPubKey.default(),
        )

    @classmethod
    def from_raw(cls, address: SolPubKey, raw: _RawAccount) -> Self:
        if raw is None:
            return cls.default()
        elif isinstance(raw, SolAccountModel):
            return raw
        elif isinstance(raw, _acct.Account):
            return cls(
                address=address,
                balance=raw.lamports,
                data=raw.data,
                owner=raw.owner,
                executable=raw.executable,
                rent_epoch=raw.rent_epoch,
            )
        raise ValueError(f"Wrong input type: {type(raw).__name__}")

    @property
    def is_empty(self) -> None:
        return self.address.is_empty


_RawAccount = Union[SolAccountModel, _acct.Account, None]
