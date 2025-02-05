from __future__ import annotations

import copy
import itertools
import logging
import time
import typing
from dataclasses import dataclass, field
from typing import Iterator, Generator, ClassVar, Sequence

from pydantic import Field
from typing_extensions import Self

from common.config.config import Config
from common.ethereum.bin_str import EthBinStrField, EthBinStr
from common.ethereum.hash import EthTxHash, EthAddress, EthTxHashField, EthBlockHash, EthHash32
from common.neon.block import NeonBlockHdrModel
from common.neon.evm_log_decoder import NeonTxEventModel, NeonTxLogReturnInfo
from common.neon.receipt_model import NeonTxReceiptModel
from common.neon.transaction_decoder import SolNeonTxMetaInfo, SolNeonTxIxMetaInfo, SolNeonAltTxIxModel
from common.neon.transaction_model import NeonTxModel
from common.solana.block import SolRpcBlockInfo
from common.solana.commit_level import SolCommit
from common.solana.pubkey import SolPubKey, SolPubKeyField
from common.solana.signature import SolTxSig
from common.solana.transaction_decoder import SolTxMetaInfo, SolTxCostModel
from common.utils.cached import cached_method, reset_cached_method, cached_property
from common.utils.format import str_fmt_object
from common.utils.pydantic import BaseModel

_LOG = logging.getLogger(__name__)


class BaseNeonIndexedObjInfo:
    class InitData(BaseModel):
        start_slot: int
        last_slot: int
        is_stuck: bool

        # protected:
        _default: ClassVar[BaseNeonIndexedObjInfo.InitData | None] = None

        @classmethod
        def default(cls) -> Self:
            if not cls._default:
                cls._default = cls(start_slot=0, last_slot=0, is_stuck=False)
            return cls._default

    def __init__(self, init=InitData.default()) -> None:
        self._start_slot = init.start_slot
        self._last_slot = init.last_slot
        self._is_stuck = init.is_stuck

        # for debugging purposes:
        #   it is useful for checking in the Indexer logs
        #   that the object is built from the correct parent
        self._prev_slot = init.last_slot

    def to_string(self) -> str:
        return str_fmt_object(self, False)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    @property
    def start_slot(self) -> int:
        return self._start_slot

    @property
    def last_slot(self) -> int:
        return self._last_slot

    @property
    def is_stuck(self) -> bool:
        return self._is_stuck

    def mark_stuck(self) -> None:
        if not self._is_stuck:
            _LOG.warning("stuck: %s", self)
        self._is_stuck = True

    def add_sol_neon_ix(self, sol_neon_ix: SolNeonTxIxMetaInfo) -> None:
        self._set_start_slot(sol_neon_ix.slot)
        self._set_last_slot(sol_neon_ix.slot)

    # protected:

    def _set_start_slot(self, slot: int) -> None:
        if self._start_slot == 0 or slot < self._start_slot:
            self._start_slot = slot

    def _set_last_slot(self, slot: int) -> None:
        if slot > self._last_slot:
            self._prev_slot = self._last_slot
            self._last_slot = slot


class NeonIndexedHolderInfo(BaseNeonIndexedObjInfo):
    @dataclass(frozen=True)
    class DataChunk:
        offset: int
        length: int
        data: bytes

        @classmethod
        def default(cls) -> Self:
            return cls(offset=0, length=0, data=bytes())

        @cached_method
        def to_string(self) -> str:
            return str_fmt_object(dict(offset=self.offset, length=self.length))

        def __str__(self) -> str:
            return self.to_string()

        def __repr__(self) -> str:
            return self.to_string()

        @property
        def is_valid(self) -> bool:
            return (self.length > 0) and (len(self.data) == self.length)

    class Key:
        def __init__(self, address: SolPubKey, neon_tx_hash: EthTxHash) -> None:
            self._addr = address
            self._neon_tx_hash = neon_tx_hash

        def __deepcopy__(self, memo: dict) -> Self:
            """The object is not mutable, so there is no point in creating a copy."""
            memo[id(self)] = self
            return self

        @classmethod
        def from_raw(cls, address: SolPubKey, neon_tx_hash: EthTxHash) -> Self:
            return cls(address, neon_tx_hash)

        def to_string(self) -> str:
            return f"{self._addr}:{self._neon_tx_hash}"

        def __str__(self) -> str:
            return self.to_string()

        def __repr__(self) -> str:
            return self.to_string()

        @cached_method
        def __hash__(self) -> int:
            return hash(tuple([self._addr, self._neon_tx_hash]))

        def __eq__(self, other) -> bool:
            if other is self:
                return True
            elif not isinstance(other, self.__class__):
                return False
            return (self._addr, self._neon_tx_hash) == (other._addr, other._neon_tx_hash)  # noqa

        @property
        def address(self) -> SolPubKey:
            return self._addr

        @property
        def neon_tx_hash(self) -> EthTxHash:
            return self._neon_tx_hash

    class InitData(BaseNeonIndexedObjInfo.InitData):
        neon_tx_hash: EthTxHashField
        address: SolPubKeyField
        data_size: int
        data: EthBinStrField

    def __init__(self, key: Key, data: bytes, data_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._key = key
        self._data_size = data_size
        self._data = data

    def __deepcopy__(self, memo: dict) -> Self:
        new_self = copy.copy(self)
        memo[id(self)] = new_self
        return new_self

    @classmethod
    def from_raw(cls, key: Key) -> Self:
        return cls(
            key=key,
            # default:
            data=bytes(),
            data_size=0,
        )

    @classmethod
    def from_dict(cls, dict_data: dict) -> Self:
        init = cls.InitData.from_dict(dict_data)

        holder = cls(
            key=cls.Key(init.address, init.neon_tx_hash),
            data=init.data.to_bytes(),
            data_size=init.data_size,
            init=init,
        )
        return holder

    def to_dict(self) -> dict:
        return self.InitData(
            start_slot=self._start_slot,
            last_slot=self._last_slot,
            is_stuck=self._is_stuck,
            neon_tx_hash=self._key.neon_tx_hash,
            address=self._key.address,
            data_size=self._data_size,
            data=self._data,
        ).to_dict()

    @property
    def key(self) -> Key:
        return self._key

    @property
    def neon_tx_hash(self) -> EthTxHash:
        return self._key.neon_tx_hash

    @property
    def address(self) -> SolPubKey:
        return self._key.address

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def data_size(self) -> int:
        return self._data_size

    def add_data_chunk(self, chunk: DataChunk) -> None:
        end_pos = chunk.offset + chunk.length
        data_len = len(self._data)
        if end_pos > data_len:
            self._data += bytes(end_pos - data_len)

        self._data = self._data[: chunk.offset] + chunk.data + self._data[end_pos:]
        self._data_size += chunk.length


@dataclass
class _NeonTxEventDraft:
    event_type: NeonTxEventModel.Type
    is_hidden: bool

    neon_tx_hash: EthTxHash
    sol_tx_sig: SolTxSig
    sol_ix_idx: int
    sol_inner_ix_idx: int | None

    address: EthAddress = EthAddress.default()
    topic_list: list[EthHash32] = field(default_factory=list)
    data: EthBinStrField = EthBinStr.default()

    total_gas_used: int = 0
    total_step_cnt: int = 0
    is_reverted: bool = False
    event_level: int = 0
    event_order: int = 0

    block_hash: EthBlockHash = EthBlockHash.default()
    slot: int = 0
    neon_tx_idx: int = 0
    block_log_idx: int | None = None
    neon_tx_log_idx: int | None = None

    def __deepcopy__(self, memo: dict) -> Self:
        new_self = copy.copy(self)
        memo[id(self)] = new_self
        return new_self

    @classmethod
    def from_raw(cls, src: NeonTxEventModel) -> Self:
        return cls(**dict(src))

    def to_clean_copy(self) -> NeonTxEventModel:
        return NeonTxEventModel.model_validate(self, from_attributes=True)

    @cached_property
    def is_exit_event_type(self) -> bool:
        return self.event_type in (
            NeonTxEventModel.Type.ExitStop,
            NeonTxEventModel.Type.ExitReturn,
            NeonTxEventModel.Type.ExitSelfDestruct,
            NeonTxEventModel.Type.ExitRevert,
            NeonTxEventModel.Type.ExitSendAll,
        )

    @cached_property
    def is_start_event_type(self) -> bool:
        return self.event_type in (
            NeonTxEventModel.Type.EnterCall,
            NeonTxEventModel.Type.EnterCallCode,
            NeonTxEventModel.Type.EnterStaticCall,
            NeonTxEventModel.Type.EnterDelegateCall,
            NeonTxEventModel.Type.EnterCreate,
            NeonTxEventModel.Type.EnterCreate2,
        )

    @property
    def is_step_reset(self) -> bool:
        return self.event_type == NeonTxEventModel.Type.StepReset

    def set_log_idx(self, block_log_idx: int, neon_tx_log_idx: int) -> None:
        self.block_log_idx = block_log_idx
        self.neon_tx_log_idx = neon_tx_log_idx

    def to_string(self) -> str:
        return str_fmt_object(self)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()


@dataclass
class _NeonTxReceiptDraft:
    slot: int | None
    block_hash: EthBlockHash

    sol_tx_sig: SolTxSig
    sol_ix_idx: int | None
    sol_inner_ix_idx: int | None

    neon_tx_idx: int | None
    status: int

    total_gas_used: int
    sum_gas_used: int
    priority_fee_spent: int

    event_list: list[NeonTxEventModel]

    is_completed: bool
    is_canceled: bool

    @classmethod
    def from_raw(cls, src: NeonTxReceiptModel) -> Self:
        self = cls(**dict(src))
        self.event_list = src.event_list.copy()
        return self

    def to_clean_copy(self) -> NeonTxReceiptModel:
        return NeonTxReceiptModel.model_validate(self, from_attributes=True)

    def set_event_list(self, event_list: list[NeonTxEventModel]) -> None:
        self.event_list = event_list

    def to_string(self) -> str:
        return str_fmt_object(self)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()


class NeonIndexedTxInfo(BaseNeonIndexedObjInfo):
    Key = EthTxHash

    class InitData(BaseNeonIndexedObjInfo.InitData):
        neon_tx_hash: EthTxHashField
        holder_address: SolPubKeyField
        chain_id: int = Field(default=0)
        operator: SolPubKeyField
        gas_used: int
        total_gas_used: int
        total_priority_fee: int = 0
        has_truncated_log: bool
        has_good_ix: bool = Field(default=False)
        neon_tx: NeonTxModel
        neon_tx_event_list: list[NeonTxEventModel]
        neon_tx_rcpt: NeonTxReceiptModel
        alt_address_list: list[SolPubKeyField] = Field(default_factory=list)

    def __init__(
        self,
        key: Key,
        chain_id: int,
        neon_tx: NeonTxModel,
        holder_address: SolPubKey,
        operator: SolPubKey,
        neon_tx_rcpt: NeonTxReceiptModel,
        gas_used: int,
        total_gas_used: int,
        total_priority_fee: int,
        has_truncated_log: bool,
        has_good_ix: bool,
        alt_address_list: list[SolPubKey],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._key = key
        self._chain_id = chain_id
        self._neon_tx = neon_tx
        self._neon_tx_rcpt = _NeonTxReceiptDraft.from_raw(neon_tx_rcpt)
        self._holder_addr = holder_address
        self._operator = operator
        self._gas_used = gas_used
        self._total_gas_used = total_gas_used
        self._total_priority_fee = total_priority_fee
        self._has_truncated_log = has_truncated_log
        self._has_good_ix = has_good_ix
        self._alt_addr_list = alt_address_list

        # default:
        self._is_done = False
        self._is_cloned = False
        self._neon_tx_event_dict: dict[int, list[_NeonTxEventDraft]] = dict()
        self._neon_tx_ret: NeonTxLogReturnInfo | None = None
        self._clean_neon_tx_rcpt: NeonTxReceiptModel | None = None

    def __deepcopy__(self, memo: dict) -> Self:
        new_self = copy.copy(self)
        new_self._is_cloned = True
        memo[id(self)] = new_self
        return new_self

    def _complete_clone(self) -> None:
        if not self._is_cloned:
            return

        self._is_cloned = False
        self._neon_tx_rcpt = copy.copy(self._neon_tx_rcpt)
        self._alt_addr_list = copy.copy(self._alt_addr_list)
        self._neon_tx_event_dict = copy.deepcopy(self._neon_tx_event_dict)

    @classmethod
    def from_raw(cls, key: Key, neon_tx: NeonTxModel, holder_address: SolPubKey) -> Self:
        return cls(
            key=key,
            chain_id=0,
            neon_tx=neon_tx,
            holder_address=holder_address,
            # default:
            operator=SolPubKey.default(),
            neon_tx_rcpt=NeonTxReceiptModel.default(),
            gas_used=0,
            total_gas_used=0,
            total_priority_fee=0,
            has_truncated_log=False,
            has_good_ix=False,
            alt_address_list=list(),
        )

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        # TODO: remove after upgrade
        neon_tx: dict = data.get("neon_tx")
        if "gas_price_legacy" in neon_tx:
            neon_tx["gas_price"] = neon_tx.pop("gas_price_legacy")

        neon_tx_rcpt: dict | None = data.get("neon_tx_rcpt", None)
        if "priority_fee_spent" not in neon_tx_rcpt:
            neon_tx_rcpt["priority_fee_spent"] = 0

        neon_tx: dict | None = data.get("neon_tx", None)
        if "tx_chain_id" in neon_tx:
            neon_tx["chain_id"] = neon_tx.pop("tx_chain_id", None)
        #
        init = cls.InitData.from_dict(data)

        self = cls(
            key=init.neon_tx_hash,
            chain_id=init.chain_id,
            neon_tx=init.neon_tx,
            holder_address=init.holder_address,
            operator=init.operator,
            neon_tx_rcpt=init.neon_tx_rcpt,
            gas_used=init.gas_used,
            total_gas_used=init.total_gas_used,
            total_priority_fee=init.total_priority_fee,
            has_truncated_log=init.has_truncated_log,
            has_good_ix=init.has_good_ix,
            alt_address_list=init.alt_address_list,
            init=init,
        )

        for event in init.neon_tx_event_list:
            tx_event_list = self._neon_tx_event_dict.setdefault(event.total_gas_used, list())
            tx_event_list.append(_NeonTxEventDraft.from_raw(event))
        return self

    def to_dict(self) -> dict:
        tx_event_list = [e.to_clean_copy() for e in itertools.chain.from_iterable(self._neon_tx_event_dict.values())]
        return self.InitData(
            start_slot=self._start_slot,
            last_slot=self._last_slot,
            is_stuck=self._is_stuck,
            neon_tx_hash=self._key,
            chain_id=self.chain_id,
            holder_address=self._holder_addr,
            operator=self._operator,
            gas_used=self._gas_used,
            total_gas_used=self._total_gas_used,
            total_priority_fee=self._total_priority_fee,
            has_truncated_log=self._has_truncated_log,
            has_good_ix=self._has_good_ix,
            neon_tx=self._neon_tx,
            neon_tx_rcpt=self._neon_tx_rcpt.to_clean_copy(),
            neon_tx_event_list=tx_event_list,
            alt_address_list=self._alt_addr_list,
        ).to_dict()

    @property
    def holder_address(self) -> SolPubKey:
        return self._holder_addr

    @property
    def neon_tx_hash(self) -> EthTxHash:
        return self._key

    @property
    def chain_id(self) -> int:
        return self._chain_id or self._neon_tx.chain_id or 0

    @property
    def key(self) -> NeonIndexedTxInfo.Key:
        return self._key

    @property
    def neon_tx(self) -> NeonTxModel:
        return self._neon_tx

    @property
    def neon_tx_rcpt(self) -> NeonTxReceiptModel:
        return self._clean_neon_tx_rcpt

    @property
    def operator(self) -> SolPubKey:
        return self._operator

    @property
    def total_gas_used(self) -> int:
        return self._total_gas_used

    @property
    def is_done(self) -> bool:
        """Return true if indexer found the receipt for the tx"""
        return self._is_done

    @property
    def is_corrupted(self) -> bool:
        """Return true if indexer didn't find all instructions for the tx"""
        return (self._neon_tx.gas_limit <= 0) or (self._gas_used != self._total_gas_used) or self._has_truncated_log

    @property
    def has_good_ix(self) -> bool:
        return self._has_good_ix

    @property
    def is_completed(self) -> bool:
        return self._neon_tx_rcpt.is_completed

    def set_chain_id(self, chain_id: int) -> None:
        self._chain_id = chain_id

    def add_alt_address(self, alt_address: SolPubKey) -> None:
        self._complete_clone()
        self._alt_addr_list.append(alt_address)

    def mark_done(self, slot: int) -> None:
        self._is_done = True
        self._set_last_slot(slot)

    def set_tx_cancel_return(self, sol_neon_ix: SolNeonTxIxMetaInfo) -> None:
        tx_ret = NeonTxLogReturnInfo(
            event_type=NeonTxEventModel.Type.Cancel,
            total_gas_used=self._total_gas_used,
        )
        self.set_tx_return(sol_neon_ix, tx_ret)

    def set_tx_lost_return(self, sol_neon_ix: SolNeonTxIxMetaInfo) -> None:
        tx_ret = NeonTxLogReturnInfo(
            event_type=NeonTxEventModel.Type.Lost,
            total_gas_used=self._total_gas_used,
        )
        self.set_tx_return(sol_neon_ix, tx_ret)

    def set_tx_return(self, sol_neon_ix: SolNeonTxIxMetaInfo, tx_return: NeonTxLogReturnInfo) -> None:
        if self._neon_tx_rcpt.is_completed:
            _LOG.debug("skip an surplus return: %s", tx_return)
            return

        self._complete_clone()
        rcpt = self._neon_tx_rcpt
        if tx_return.event_type == NeonTxEventModel.Type.Return:
            rcpt.is_completed = True
        elif tx_return.event_type == NeonTxEventModel.Type.Cancel:
            rcpt.is_completed = True
            rcpt.is_canceled = True

        rcpt.status = tx_return.status
        rcpt.total_gas_used = tx_return.total_gas_used
        rcpt.sol_tx_sig = sol_neon_ix.sol_tx_sig
        rcpt.sol_ix_idx = sol_neon_ix.sol_ix_idx
        rcpt.sol_inner_ix_idx = sol_neon_ix.sol_inner_ix_idx

        self._neon_tx_ret = tx_return

    def set_neon_tx(self, neon_tx: NeonTxModel, holder: NeonIndexedHolderInfo) -> None:
        # assert not self._neon_tx.is_valid
        # assert neon_tx.is_valid

        self._neon_tx = neon_tx
        self._set_start_slot(holder.start_slot)
        self._set_last_slot(holder.last_slot)

    def add_sol_neon_ix(self, sol_neon_ix: SolNeonTxIxMetaInfo) -> None:
        super().add_sol_neon_ix(sol_neon_ix)
        self._gas_used += sol_neon_ix.neon_tx_ix_gas_used
        if sol_neon_ix.neon_total_gas_used > self._total_gas_used:
            self._total_gas_used = sol_neon_ix.neon_total_gas_used

        self._total_priority_fee += sol_neon_ix.neon_tx_ix_priority_fee

        if self._operator.is_empty:
            self._operator = sol_neon_ix.operator

    def extend_neon_tx_event_list(self, sol_neon_ix: SolNeonTxIxMetaInfo) -> None:
        if not (tx_event_list := [_NeonTxEventDraft.from_raw(e) for e in sol_neon_ix.iter_neon_tx_event]):
            return

        total_gas_used = sol_neon_ix.neon_total_gas_used or self._total_gas_used
        if not sol_neon_ix.is_success:
            event_hide_info = True, True, total_gas_used
            for event in tx_event_list:
                event.is_reverted, event.is_hidden, event.total_gas_used = event_hide_info
        elif sol_neon_ix.is_log_truncated:
            self._has_truncated_log = True
        else:
            self._has_good_ix = True

        self._complete_clone()
        self._neon_tx_event_dict.setdefault(total_gas_used, list()).extend(tx_event_list)

    def complete_neon_tx_event_list(
        self,
        neon_block_hdr: NeonBlockHdrModel,
        neon_tx_idx: int,
        start_log_idx: int,
        sum_gas_used: int,
    ) -> tuple[int, int]:
        assert not self._clean_neon_tx_rcpt
        assert not self.is_corrupted

        sum_gas_used += self._total_gas_used
        if not self._neon_tx_event_dict:
            return len(self._neon_tx_rcpt.event_list), sum_gas_used

        self._complete_clone()
        rcpt = self._neon_tx_rcpt
        rcpt.slot = neon_block_hdr.slot
        rcpt.block_hash = neon_block_hdr.block_hash
        rcpt.neon_tx_idx = neon_tx_idx
        rcpt.sum_gas_used = sum_gas_used
        rcpt.priority_fee_spent = self._total_priority_fee

        neon_tx_event_list = self._get_sorted_tx_event_list()
        self._fill_tx_event_order_nums(neon_tx_event_list)
        if self.is_corrupted:
            return 0, 0

        self._hide_reverted_tx_events(neon_tx_event_list)
        self._add_tx_return_event(neon_tx_event_list)
        last_log_idx = self._fill_tx_event_block_info(start_log_idx, neon_tx_event_list)

        rcpt.set_event_list([e.to_clean_copy() for e in neon_tx_event_list])
        self._neon_tx_event_dict.clear()

        self._clean_neon_tx_rcpt = rcpt.to_clean_copy()

        return last_log_idx, sum_gas_used

    # protected:

    def _get_sorted_tx_event_list(self) -> list[_NeonTxEventDraft]:
        # no events
        if not self._neon_tx_event_dict:
            return list()

        # The first step: sort dictionary by the total gas:
        #    source = {
        #        20: [event21, event22],  <-- total_gas = 20
        #        10: [event11, event12],  <-- total_gas = 10
        #        30: [event31, event32],  <-- total_gas = 30
        #    }
        #    result = [
        #        10: [event11, event12],  <-- total_gas = 10
        #        20: [event21, event22],  <-- total_gas = 20
        #        30: [event31, event32],  <-- total_gas = 30
        #    ]
        sorted_event_ll = [event_l for k, event_l in sorted(self._neon_tx_event_dict.items(), key=lambda x: x[0])]
        # The second step: combine all lists into one:
        #    result = [
        #        event11, event12,  <-- total_gas = 10
        #        event21, event22,  <-- total_gas = 20
        #        event31, event32,  <-- total_gas = 30
        #    ]
        return list(itertools.chain.from_iterable(sorted_event_ll))

    def _fill_tx_event_order_nums(self, neon_tx_event_list: list[_NeonTxEventDraft]) -> None:
        current_level, current_order, total_step_cnt = 0, 0, 0
        addr_stack: list[EthAddress] = list()

        for event in neon_tx_event_list:
            # events from broken iterations
            if event.is_reverted:
                continue

            is_tx_restart = False
            if event.is_step_reset:
                is_tx_restart = True

            #  iteration 1 (step 10)
            #  iteration 2 (step 20)
            #  iteration 3 (step 10) <--- we are here: the place of tx-restart
            #  iteration 4 (step 20)
            elif event.total_step_cnt < total_step_cnt:
                is_tx_restart = True

            if is_tx_restart:
                current_level, current_order = 0, 0
                addr_stack.clear()

            total_step_cnt = event.total_step_cnt

            if event.is_start_event_type:
                current_level += 1
                event_level, addr = current_level, event.address
                addr_stack.append(addr)
            elif event.is_exit_event_type:
                if not addr_stack:
                    _LOG.debug("bad %s in %s", event.event_type.name, self.neon_tx_hash)
                    self._has_truncated_log = True
                    break

                event_level, addr = current_level, addr_stack.pop()
                current_level -= 1
            else:
                event_level = current_level
                if addr_stack and (event.event_type != NeonTxEventModel.Type.Log):
                    addr = addr_stack[-1]
                else:
                    addr = EthAddress.default()

            current_order += 1
            event.event_level, event.event_order = event_level, current_order
            if event.address.is_empty and (not addr.is_empty):
                event.address = addr

    def _hide_reverted_tx_events(self, neon_tx_event_list: list[_NeonTxEventDraft]) -> None:
        is_failed = self._neon_tx_rcpt.status == 0
        reverted_level, is_dropped, total_step_cnt = -1, False, 2**64

        for event in reversed(neon_tx_event_list):
            if is_dropped:
                # events from restarted iterations
                is_reverted, is_hidden = True, True
            elif event.is_reverted:
                # events from broken iterations
                is_reverted, is_hidden = True, True
            elif event.is_step_reset:
                # restart of the tx execution
                is_dropped, is_reverted, is_hidden = True, True, True
            elif event.total_step_cnt > total_step_cnt:
                # tx restart:
                #   iteration 4 (step 20)
                #   iteration 3 (step 10)  <- the first iteration, all next iters were canceled
                #   iteration 2 (step 20)  <--- we are here
                #   iteration 1 (step 10)
                is_dropped, is_reverted, is_hidden = True, True, True
            else:
                if event.is_start_event_type:
                    if event.event_level == reverted_level:
                        reverted_level = -1
                elif event.is_exit_event_type:
                    if (event.event_type == NeonTxEventModel.Type.ExitRevert) and (reverted_level == -1):
                        reverted_level = event.event_level

                total_step_cnt = event.total_step_cnt
                is_reverted = (reverted_level != -1) or is_failed
                is_hidden = event.is_hidden or is_reverted

            event.is_hidden, event.is_reverted = is_hidden, is_reverted

    def _add_tx_return_event(self, neon_tx_event_list: list[_NeonTxEventDraft]) -> None:
        ret = self._neon_tx_ret
        rcpt = self._neon_tx_rcpt
        event = _NeonTxEventDraft(
            event_type=ret.event_type,
            is_hidden=True,
            neon_tx_hash=self.neon_tx_hash,
            sol_tx_sig=rcpt.sol_tx_sig,
            sol_ix_idx=rcpt.sol_ix_idx,
            sol_inner_ix_idx=rcpt.sol_inner_ix_idx,
            data=EthBinStr.from_raw(ret.status.to_bytes(1, "little")),
            total_gas_used=self._neon_tx_ret.total_gas_used,
            event_order=len(neon_tx_event_list) + 1,
        )
        neon_tx_event_list.append(event)

    def _fill_tx_event_block_info(self, start_log_idx: int, neon_tx_event_list: list[_NeonTxEventDraft]) -> int:
        rcpt = self._neon_tx_rcpt

        tx_log_idx = 0
        block_log_idx = start_log_idx
        block_info = rcpt.slot, rcpt.block_hash, rcpt.neon_tx_idx

        for event in neon_tx_event_list:
            event.slot, event.block_hash, event.neon_tx_idx = block_info
            if not event.is_hidden:
                event.block_log_idx, block_log_idx = block_log_idx, block_log_idx + 1
                event.neon_tx_log_idx, tx_log_idx = tx_log_idx, tx_log_idx + 1

        return block_log_idx


class SolIndexedAltInfo:
    Key = SolPubKey

    class InitData(BaseModel):
        key: SolPubKeyField
        neon_tx_hash: EthTxHashField
        operator: SolPubKeyField = SolPubKey.default()
        slot: int
        next_check_slot: int
        last_ix_slot: int
        is_stuck: bool

    def __init__(
        self,
        key: Key,
        neon_tx_hash: EthTxHash,
        operator: SolPubKey,
        slot: int,
        next_check_slot: int,
        last_tx_slot: int,
        is_stuck: bool,
    ) -> None:
        self._key = key
        self._neon_tx_hash = neon_tx_hash
        self._operator = operator
        self._slot = slot
        self._next_check_slot = next_check_slot
        self._last_ix_slot = last_tx_slot
        self._is_stuck = is_stuck

    @classmethod
    def from_raw(cls, key: Key, neon_tx_hash: EthTxHash, slot: int) -> Self:
        return cls(
            key=key,
            neon_tx_hash=neon_tx_hash,
            slot=slot,
            # default:
            operator=SolPubKey.default(),
            next_check_slot=0,
            last_tx_slot=0,
            is_stuck=False,
        )

    @classmethod
    def from_dict(cls, dict_data: dict) -> Self:
        init = cls.InitData.from_dict(dict_data)
        return cls(
            key=init.key,
            neon_tx_hash=init.neon_tx_hash,
            operator=init.operator,
            slot=init.slot,
            next_check_slot=init.next_check_slot,
            last_tx_slot=init.last_ix_slot,
            is_stuck=init.is_stuck,
        )

    def to_dict(self) -> dict:
        return self.InitData(
            key=self._key,
            neon_tx_hash=self._neon_tx_hash,
            operator=self._operator,
            slot=self._slot,
            next_check_slot=self._next_check_slot,
            last_ix_slot=self._last_ix_slot,
            is_stuck=self._is_stuck,
        ).to_dict()

    @property
    def key(self) -> SolPubKey:
        return self._key

    @property
    def address(self) -> SolPubKey:
        return self._key

    @property
    def neon_tx_hash(self) -> EthTxHash:
        return self._neon_tx_hash

    @property
    def operator(self) -> SolPubKey:
        return self._operator

    @property
    def slot(self) -> int:
        return self._slot

    @property
    def next_check_slot(self) -> int:
        return self._next_check_slot

    @property
    def last_sol_ix_slot(self) -> int:
        return self._last_ix_slot

    @property
    def is_stuck(self) -> bool:
        return self._is_stuck

    def to_string(self) -> str:
        return str_fmt_object(self, skip_underscore_prefix=False)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def set_next_check_slot(self, slot: int) -> None:
        if slot > self._next_check_slot:
            self._next_check_slot = slot

    def set_last_ix_slot(self, slot: int, operator: SolPubKey) -> None:
        if slot > self._last_ix_slot:
            self._last_ix_slot = slot
        if self._operator.is_empty:
            self._operator = operator

    def mark_stuck(self) -> None:
        if self.is_stuck:
            return

        self._is_stuck = True
        _LOG.warning("stuck: %s", self)


class NeonIndexedBlockInfo:
    def __init__(self, sol_block: SolRpcBlockInfo) -> None:
        self._sol_block = sol_block
        self._min_slot = sol_block.slot
        self._stuck_slot = sol_block.slot
        self._is_completed = False
        self._is_cloned = True
        self._is_done = False
        self._has_corrupted_tx = False
        self._is_stuck_completed = False

        self._neon_holder_dict: dict[NeonIndexedHolderInfo.Key, NeonIndexedHolderInfo] = dict()
        self._modified_neon_acct_set: set[SolPubKey] = set()
        self._stuck_neon_holder_list: list[NeonIndexedHolderInfo] = list()
        self._failed_neon_holder_set: set[NeonIndexedHolderInfo.Key] = set()

        self._neon_tx_dict: dict[NeonIndexedTxInfo.Key, NeonIndexedTxInfo] = dict()
        self._done_neon_tx_list: list[NeonIndexedTxInfo] = list()
        self._stuck_neon_tx_list: list[NeonIndexedTxInfo] = list()
        self._failed_neon_tx_set: set[NeonIndexedTxInfo.Key] = set()

        self._sol_alt_dict: dict[SolIndexedAltInfo.Key, SolIndexedAltInfo] = dict()

        self._sol_neon_ix_list: list[SolNeonTxIxMetaInfo] = list()
        self._sol_alt_ix_list: list[SolNeonAltTxIxModel] = list()
        self._sol_tx_cost_list: list[SolTxCostModel] = list()

    @staticmethod
    def from_block(src_block: NeonIndexedBlockInfo, sol_block: SolRpcBlockInfo) -> NeonIndexedBlockInfo:
        new_block = NeonIndexedBlockInfo(sol_block)
        new_block._is_cloned = False

        if len(src_block._neon_tx_dict) or len(src_block._neon_holder_dict):
            new_block._min_slot = src_block._min_slot

        if src_block._stuck_slot > new_block.slot:
            new_block._stuck_slot = src_block._stuck_slot

        new_block._neon_holder_dict = src_block._neon_holder_dict
        new_block._stuck_neon_holder_list = src_block._stuck_neon_holder_list
        new_block._failed_neon_holder_set = src_block._failed_neon_holder_set

        new_block._neon_tx_dict = src_block._neon_tx_dict
        new_block._stuck_neon_tx_list = src_block._stuck_neon_tx_list
        new_block._failed_neon_tx_set = src_block._failed_neon_tx_set

        new_block._sol_alt_dict = src_block._sol_alt_dict

        return new_block

    @staticmethod
    def from_stuck_data(
        sol_block: SolRpcBlockInfo,
        stuck_slot: int,
        neon_holder_list: Sequence[dict],
        neon_tx_list: Sequence[dict],
        alt_list: Sequence[dict],
    ) -> NeonIndexedBlockInfo:

        new_block = NeonIndexedBlockInfo(sol_block)
        new_block._stuck_slot = stuck_slot

        for src in neon_holder_list:
            holder = NeonIndexedHolderInfo.from_dict(src)
            new_block._neon_holder_dict[holder.key] = holder
            new_block._stuck_neon_holder_list.append(holder)

        for src in neon_tx_list:
            tx = NeonIndexedTxInfo.from_dict(src)
            new_block._neon_tx_dict[tx.key] = tx
            new_block._stuck_neon_tx_list.append(tx)

        for src in alt_list:
            alt = SolIndexedAltInfo.from_dict(src)
            new_block._sol_alt_dict[alt.key] = alt

        return new_block

    def to_string(self) -> str:
        return str_fmt_object(self, False)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    @property
    def slot(self) -> int:
        return self._sol_block.slot

    @property
    def stuck_slot(self) -> int:
        return self._stuck_slot

    @property
    def is_finalized(self) -> bool:
        return self._sol_block.is_finalized

    @property
    def is_completed(self) -> bool:
        return self._is_completed

    @property
    def is_corrupted(self) -> bool:
        return self._has_corrupted_tx

    @property
    def is_done(self) -> bool:
        return self._is_done

    @property
    def min_slot(self) -> int:
        return self._min_slot

    def mark_finalized(self) -> None:
        if self._sol_block.is_finalized:
            return
        self._sol_block.mark_finalized()
        self._to_neon_block_hdr.reset_cache(self)

    def mark_done(self) -> None:
        self._is_done = True

    def add_sol_neon_ix(self, sol_neon_ix: SolNeonTxIxMetaInfo) -> None:
        self._clone()
        self._sol_neon_ix_list.append(sol_neon_ix)

    def add_sol_tx_cost(self, sol_tx_cost: SolTxCostModel) -> None:
        self._sol_tx_cost_list.append(sol_tx_cost)

    def find_neon_tx_holder(self, address: SolPubKey, sol_neon_ix: SolNeonTxIxMetaInfo) -> NeonIndexedHolderInfo | None:
        key = NeonIndexedHolderInfo.Key.from_raw(address, sol_neon_ix.neon_tx_hash)
        if not (holder := self._neon_holder_dict.get(key, None)):
            return None

        holder.add_sol_neon_ix(sol_neon_ix)
        self._modified_neon_acct_set.add(key.address)
        return holder

    def find_or_add_neon_tx_holder(self, address: SolPubKey, sol_neon_ix: SolNeonTxIxMetaInfo) -> NeonIndexedHolderInfo:
        key = NeonIndexedHolderInfo.Key.from_raw(address, sol_neon_ix.neon_tx_hash)
        if holder := self._neon_holder_dict.get(key, None):
            return holder

        holder = NeonIndexedHolderInfo.from_raw(key)
        holder.add_sol_neon_ix(sol_neon_ix)
        self._neon_holder_dict[key] = holder
        self._modified_neon_acct_set.add(key.address)
        return holder

    def done_neon_holder(self, holder: NeonIndexedHolderInfo) -> None:
        self._del_neon_holder(holder)

    def destroy_neon_holder(self, address: SolPubKey) -> None:
        self._modified_neon_acct_set.add(address)

    def find_neon_tx(self, sol_neon_ix: SolNeonTxIxMetaInfo, *, skip_add_gas: bool = False) -> NeonIndexedTxInfo | None:
        key = NeonIndexedTxInfo.Key.from_raw(sol_neon_ix.neon_tx_hash)
        if not (tx := self._neon_tx_dict.get(key, None)):
            return None

        if not skip_add_gas:
            tx.add_sol_neon_ix(sol_neon_ix)
        self._add_alt(tx, sol_neon_ix)
        return tx

    def add_neon_tx(
        self, neon_tx: NeonTxModel, holder_address: SolPubKey, sol_neon_ix: SolNeonTxIxMetaInfo
    ) -> NeonIndexedTxInfo:
        key = NeonIndexedTxInfo.Key.from_raw(sol_neon_ix.neon_tx_hash)
        # assert key not in self._neon_tx_dict, f"the tx {key} ia already in use!"

        tx = NeonIndexedTxInfo.from_raw(key, neon_tx, holder_address)
        tx.add_sol_neon_ix(sol_neon_ix)
        self._neon_tx_dict[key] = tx
        self._add_alt(tx, sol_neon_ix)
        return tx

    def done_neon_tx(self, tx: NeonIndexedTxInfo) -> None:
        if tx.is_done:
            _LOG.error("attempt to done the completed tx %s", tx)
            return

        tx.mark_done(self.slot)
        if tx.has_good_ix:
            self._done_neon_tx_list.append(tx)
        else:
            self._del_neon_tx(tx)

    def done_alt(self, alt: SolIndexedAltInfo) -> None:
        self._sol_alt_dict.pop(alt.key)

    def add_alt_ix(self, alt: SolIndexedAltInfo, alt_ix: SolNeonAltTxIxModel) -> None:
        self._sol_tx_cost_list.append(alt_ix.sol_tx_cost)
        self._sol_alt_ix_list.append(alt_ix)
        alt.set_last_ix_slot(alt_ix.slot, alt_ix.sol_tx_cost.sol_signer)

    def iter_stuck_neon_holder(self) -> Iterator[NeonIndexedHolderInfo]:
        # assert self._is_stuck_completed
        return iter(self._stuck_neon_holder_list)

    def iter_stuck_neon_tx(self) -> Iterator[NeonIndexedTxInfo]:
        # assert self._is_stuck_completed
        return iter(self._stuck_neon_tx_list)

    def fail_neon_holder_list(self, failed_holder_list: list[NeonIndexedHolderInfo]) -> None:
        failed_holder_set: set[NeonIndexedHolderInfo.Key] = set()
        for holder in failed_holder_list:
            if holder.key not in self._failed_neon_holder_set:
                failed_holder_set.add(holder.key)
                continue

            # Remove holder only if it appears two times in the failed set
            _LOG.warning("skip lost: %s", holder)
            self._del_neon_holder(holder)

        # Replace old set with the new one - so it is not required to clone it
        self._failed_neon_holder_set = failed_holder_set

    def fail_neon_tx_list(self, failed_tx_list: list[NeonIndexedTxInfo]) -> None:
        failed_tx_set: set[NeonIndexedTxInfo.Key] = set()
        for tx in failed_tx_list:
            if tx.key not in self._failed_neon_tx_set:
                failed_tx_set.add(tx.key)
                continue

            # Remove the tx only if it appears two times in the failed set
            _LOG.warning("skip lost: %s", tx)
            self._del_neon_tx(tx)

        # Replace old set with the new one - so it is not required to clone it
        self._failed_neon_tx_set = failed_tx_set

    @property
    def neon_block_hdr(self) -> NeonBlockHdrModel:
        return self._to_neon_block_hdr()

    @property
    def neon_tx_cnt(self) -> int:
        return len(self._neon_tx_dict)

    @property
    def neon_holder_cnt(self) -> int:
        return len(self._neon_holder_dict)

    @property
    def sol_neon_ix_cnt(self) -> int:
        return len(self._sol_neon_ix_list)

    @property
    def sol_alt_cnt(self) -> int:
        return len(self._sol_alt_dict)

    def iter_sol_neon_ix(self) -> Iterator[SolNeonTxIxMetaInfo]:
        return iter(self._sol_neon_ix_list)

    def iter_sol_alt_ix(self) -> Iterator[SolNeonAltTxIxModel]:
        return iter(self._sol_alt_ix_list)

    def iter_sol_tx_cost(self) -> Iterator[SolTxCostModel]:
        return iter(self._sol_tx_cost_list)

    def iter_done_neon_tx(self) -> Iterator[NeonIndexedTxInfo]:
        if self._has_corrupted_tx:
            # don't override db with corrupted data
            return iter(())
        return iter(self._done_neon_tx_list)

    def iter_alt(self) -> Iterator[SolIndexedAltInfo]:
        return iter(self._sol_alt_dict.values())

    def complete_block(self) -> None:
        assert not self._is_completed
        self._is_completed = True
        self._finalize_log_list()

    def check_stuck_objs(self, cfg: Config) -> None:
        if self._is_stuck_completed:
            return

        self._is_stuck_completed = True
        self._check_stuck_holders(cfg)
        self._check_stuck_txs(cfg)
        self._check_stuck_alts(cfg)

    # protected:

    def _clone(self) -> None:
        if self._is_cloned:
            return

        slot = self.slot
        self._is_cloned = True
        self._min_slot = slot
        if self._stuck_slot < slot:
            self._stuck_slot = slot

        self._neon_holder_dict = copy.deepcopy(self._neon_holder_dict)
        self._neon_tx_dict = copy.deepcopy(self._neon_tx_dict)
        self._sol_alt_dict = copy.deepcopy(self._sol_alt_dict)

    @reset_cached_method
    def _to_neon_block_hdr(self) -> NeonBlockHdrModel:
        return NeonBlockHdrModel.from_raw(self._sol_block)

    def _del_neon_holder(self, holder: NeonIndexedHolderInfo) -> None:
        if not self._neon_holder_dict.pop(holder.key, None):
            _LOG.error("attempt to remove the not-existing %s", holder)

    def _add_alt(self, tx: NeonIndexedTxInfo, sol_neon_ix: SolNeonTxIxMetaInfo) -> None:
        for alt_addr in sol_neon_ix.iter_alt_address():
            if alt_addr in self._sol_alt_dict:
                continue
            alt = SolIndexedAltInfo.from_raw(alt_addr, tx.neon_tx_hash, sol_neon_ix.slot)
            tx.add_alt_address(alt_addr)
            self._sol_alt_dict[alt_addr] = alt

    def _del_neon_tx(self, tx: NeonIndexedTxInfo) -> None:
        if not self._neon_tx_dict.pop(tx.key, None):
            _LOG.error("attempt to remove the not-exist %s", tx)

    def _finalize_log_list(self) -> None:
        log_idx = 0
        tx_idx = 0
        sum_gas_used = 0
        neon_block_hdr = self.neon_block_hdr
        for tx in self._done_neon_tx_list:
            self._del_neon_tx(tx)
            if tx.is_corrupted:
                _LOG.error("corrupted tx: %s", tx)
                self._has_corrupted_tx = True
                continue
            elif self._has_corrupted_tx:
                _LOG.warning("block is corrupted, skip tx: %s", tx)
                continue

            log_idx, sum_gas_used = tx.complete_neon_tx_event_list(neon_block_hdr, tx_idx, log_idx, sum_gas_used)
            if tx.is_corrupted:
                _LOG.error("corrupted tx: %s", tx)
                self._has_corrupted_tx = True
                continue

            tx_idx += 1

    def _check_stuck_holders(self, cfg: Config) -> None:
        slot = self.slot
        # there were the restart with stuck holders
        if self._stuck_slot > slot:
            return
        # if was no changes
        elif not self._is_cloned:
            # if all holders are already stuck
            if len(self._stuck_neon_holder_list) == len(self._neon_holder_dict):
                return

        stuck_slot = slot - cfg.stuck_object_blockout
        self._stuck_neon_holder_list = list()

        for holder in list(self._neon_holder_dict.values()):
            if (holder.last_slot < slot) and (holder.address in self._modified_neon_acct_set):
                _LOG.debug("on block-slot %s skip the stuck: %s", slot, holder)
                self._del_neon_holder(holder)

            elif stuck_slot > holder.start_slot:
                self._stuck_neon_holder_list.append(holder)
                holder.mark_stuck()

            elif self._min_slot > holder.start_slot:
                self._min_slot = holder.start_slot

        self._modified_neon_acct_set.clear()

    def _check_stuck_txs(self, cfg: Config) -> None:
        # there were the restart with stuck txs
        slot = self.slot
        if self._stuck_slot > slot:
            return
        # if was no changes
        elif not self._is_cloned:
            # if all txs are already stuck
            if len(self._stuck_neon_tx_list) == len(self._neon_tx_dict):
                return

        stuck_slot = slot - cfg.stuck_object_blockout
        self._stuck_neon_tx_list = list()

        for tx in list(self._neon_tx_dict.values()):
            if tx.is_done:
                continue

            elif stuck_slot > tx.start_slot:
                self._stuck_neon_tx_list.append(tx)
                tx.mark_stuck()

            elif self._min_slot > tx.start_slot:
                self._min_slot = tx.start_slot

    def _check_stuck_alts(self, cfg: Config) -> None:
        slot = self.slot
        stuck_slot = slot - cfg.alt_freeing_depth
        if stuck_slot < 0:
            return

        for alt in self._sol_alt_dict.values():
            if stuck_slot > alt.slot:
                alt.mark_stuck()
            if self._min_slot > alt.slot:
                self._min_slot = alt.slot


class NeonIndexedBlockDict:
    def __init__(self):
        self._neon_block_dict: dict[int, NeonIndexedBlockInfo] = dict()
        self._finalized_neon_block: NeonIndexedBlockInfo | None = None
        self._min_slot = 0

    @property
    def finalized_neon_block(self) -> NeonIndexedBlockInfo | None:
        return self._finalized_neon_block

    @property
    def min_slot(self) -> int:
        return self._min_slot

    def clear(self):
        self._neon_block_dict.clear()
        self._finalized_neon_block = None
        self._min_slot = 0

    def find_neon_block(self, slot: int) -> NeonIndexedBlockInfo | None:
        return self._neon_block_dict.get(slot, None)

    def add_neon_block(self, neon_block: NeonIndexedBlockInfo) -> None:
        if neon_block.slot in self._neon_block_dict:
            return

        self._neon_block_dict[neon_block.slot] = neon_block
        # _LOG.debug("add block %s", neon_block.slot)

    def finalize_neon_block(self, neon_block: NeonIndexedBlockInfo) -> None:
        assert neon_block.slot in self._neon_block_dict

        # remove all blocks before the finalized slot
        #   the container has only one finalized block
        if self._finalized_neon_block is not None:
            for slot in range(self._finalized_neon_block.slot, neon_block.slot):
                self._neon_block_dict.pop(slot, None)

        # _LOG.debug("finalize block %s", neon_block.slot)
        self._finalized_neon_block = neon_block
        self._min_slot = neon_block.min_slot


class SolNeonDecoderStat:
    sol_tx_meta_cnt: int = 0
    sol_neon_ix_cnt: int = 0
    sol_block_cnt: int = 0
    neon_corrupted_block_cnt: int = 0
    _prev_neon_corrupted_block_cnt: int = 0

    _in_process: bool = False
    _start_time: int = 0
    _total_time: int = 0

    def reset(self) -> None:
        self._start_time = time.monotonic_ns()
        self._total_time = 0
        self.sol_neon_ix_cnt = 0
        self.sol_tx_meta_cnt = 0
        self.sol_block_cnt = 0
        self.neon_corrupted_block_cnt = 0
        self._prev_neon_corrupted_block_cnt = 0

    def start_timer(self) -> None:
        self.commit_timer()
        self._in_process = True
        self._start_time = time.monotonic_ns()

    def commit_timer(self) -> None:
        if self._in_process:
            self._total_time = self.processing_time_msec
            self._in_process = False

    @property
    def neon_corrupted_block_cnt_diff(self) -> int:
        value = self.neon_corrupted_block_cnt - self._prev_neon_corrupted_block_cnt
        self._prev_neon_corrupted_block_cnt = self.neon_corrupted_block_cnt
        return value

    @property
    def processing_time_msec(self) -> int:
        time_diff = self._total_time
        if self._in_process:
            time_diff += (time.monotonic_ns() - self._start_time) // (10**6)
        return time_diff

    def inc_sol_neon_ix_cnt(self) -> None:
        self.sol_neon_ix_cnt += 1

    def add_sol_tx_meta_cnt(self, value: int) -> None:
        self.sol_tx_meta_cnt += value

    def inc_sol_block_cnt(self) -> None:
        self.sol_block_cnt += 1

    def inc_neon_corrupted_block_cnt(self) -> None:
        self.neon_corrupted_block_cnt += 1


class SolNeonDecoderCtx:
    # Iterate:
    #   for solana_block in block_range(start_slot, stop_slot):
    #       for solana_tx in solana_block.solana_tx_list:
    #           for solana_ix in solana_tx.solana_ix_list:
    #               solana_ix.level <- level in stack of calls
    #  ....
    def __init__(self, cfg: Config, stat: SolNeonDecoderStat):
        self._cfg = cfg
        self._stat = stat

        self._start_slot = 0
        self._stop_slot = 0
        self._sol_commit = SolCommit.Processed
        self._is_finalized = False

        self._sol_tx_meta: SolTxMetaInfo | None = None
        self._sol_neon_ix: SolNeonTxIxMetaInfo | None = None

        self._neon_block: NeonIndexedBlockInfo | None = None
        self._neon_block_queue: list[NeonIndexedBlockInfo] = list()

    def to_string(self) -> str:
        return str_fmt_object(
            dict(
                start_slot=self._start_slot,
                stop_slot=self._stop_slot,
                sol_commit=self._sol_commit,
                slot=self._neon_block.slot if self.has_neon_block else None,
            )
        )

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def set_slot_range(self, start_slot: int, stop_slot: int, sol_commit: SolCommit) -> None:
        self._start_slot = start_slot
        self._stop_slot = stop_slot
        self._sol_commit = sol_commit
        self._is_finalized = sol_commit == SolCommit.Finalized

    def set_neon_block(self, neon_block: NeonIndexedBlockInfo) -> None:
        self._neon_block = neon_block

    @property
    def start_slot(self) -> int:
        return self._start_slot

    @property
    def stop_slot(self) -> int:
        return self._stop_slot

    @property
    def sol_commit(self) -> SolCommit:
        return self._sol_commit

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    @property
    def neon_block(self) -> NeonIndexedBlockInfo:
        assert self._neon_block is not None
        return self._neon_block

    @property
    def has_neon_block(self) -> bool:
        return self._neon_block is not None

    def add_neon_block_to_queue(self) -> None:
        def _last_neon_slot() -> int:
            if not self._neon_block_queue:
                return self._start_slot - 1
            return self._neon_block_queue[-1].slot

        neon_block = self._neon_block
        assert neon_block.slot > _last_neon_slot()
        self._neon_block_queue.append(neon_block)

        is_pre_stuck_block = neon_block.stuck_slot > neon_block.slot
        if not is_pre_stuck_block:
            if (not neon_block.is_done) and neon_block.is_corrupted:
                self._stat.inc_neon_corrupted_block_cnt()

    @property
    def is_neon_block_queue_empty(self) -> bool:
        return len(self._neon_block_queue) == 0

    @property
    def is_neon_block_queue_full(self) -> bool:
        return self._cfg.indexer_poll_block_cnt <= len(self._neon_block_queue)

    def clear_neon_block_queue(self) -> None:
        self._neon_block_queue.clear()

    def iter_sol_neon_tx_meta(self, sol_block: SolRpcBlockInfo) -> Generator[SolTxMetaInfo, None, None]:
        try:
            self._stat.inc_sol_block_cnt()
            self._stat.add_sol_tx_meta_cnt(len(sol_block.tx_list))
            for sol_tx in sol_block.tx_list:
                self._sol_tx_meta = SolTxMetaInfo.from_raw(sol_block.slot, sol_tx)
                yield self._sol_tx_meta
        finally:
            self._sol_tx_meta = None

    @property
    def sol_neon_ix(self) -> SolNeonTxIxMetaInfo:
        # assert self._sol_neon_ix is not None
        return typing.cast(SolNeonTxIxMetaInfo, self._sol_neon_ix)

    def iter_sol_neon_ix(self) -> Generator[SolNeonTxIxMetaInfo, None, None]:
        # assert self._sol_tx_meta is not None

        try:
            sol_neon_tx = SolNeonTxMetaInfo.from_raw(self._sol_tx_meta)
            for self._sol_neon_ix in sol_neon_tx.sol_neon_ix_list():
                self._stat.inc_sol_neon_ix_cnt()
                yield self._sol_neon_ix
        finally:
            self._sol_neon_ix = None

    @property
    def neon_block_queue(self) -> Sequence[NeonIndexedBlockInfo]:
        return tuple(self._neon_block_queue)
