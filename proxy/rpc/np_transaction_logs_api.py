from __future__ import annotations

import asyncio
from typing import ClassVar, Sequence

from pydantic import Field

from common.ethereum.commit_level import EthCommit
from common.ethereum.hash import EthBlockHashField, EthAddressField, EthHash32Field, EthAddress, EthBlockHash, EthHash32
from common.jsonrpc.api import BaseJsonRpcModel
from common.jsonrpc.errors import InvalidParamError
from common.utils.cached import cached_property
from .api import RpcBlockRequest, RpcEthTxEventModel, RpcNeonTxEventModel
from .server_abc import NeonProxyApi


class _RpcLogListRequest(BaseJsonRpcModel):
    blockHash: EthBlockHashField = Field(EthBlockHash.default())
    address: EthAddressField | list[EthAddressField] = Field(EthAddress.default())
    fromBlock: RpcBlockRequest | None = None
    toBlock: RpcBlockRequest | None = None
    topicList: list[EthHash32Field | list[EthHash32Field]] = Field(default_factory=list, validation_alias="topics")

    @cached_property
    def address_list(self) -> Sequence[EthAddress]:
        if isinstance(self.address, list):
            return tuple(filter(lambda a: not a.is_empty, self.address))
        if not self.address.is_empty:
            return tuple([self.address])
        return tuple()

    @cached_property
    def topic_list(self) -> Sequence[Sequence[EthHash32]]:
        if not self.topicList:
            return tuple()

        topic_list: list[Sequence[EthHash32]] = list()
        for topic in self.topicList:
            if isinstance(topic, list):
                topic = tuple(filter(lambda t: not t.is_empty, topic))
            elif topic.is_empty:
                topic = tuple([])
            else:
                topic = tuple([topic])
            topic_list.append(topic)
        return tuple(topic_list)

    @cached_property
    def from_block(self) -> RpcBlockRequest | None:
        assert self.blockHash.is_empty

        if not self.fromBlock:
            return None
        elif self.fromBlock.is_block_name and (self.fromBlock.block_name == EthCommit.Earliest):
            return None
        elif self.fromBlock.is_block_number and (self.fromBlock.block_number == 0):
            return None
        return self.fromBlock

    @cached_property
    def to_block(self) -> RpcBlockRequest | None:
        assert self.blockHash.is_empty

        if not self.toBlock:
            return None
        elif self.toBlock.is_block_name and (self.toBlock.block_name in (EthCommit.Pending, EthCommit.Latest)):
            return None
        return self.toBlock

    def model_post_init(self, _ctx) -> None:
        if (self.fromBlock or self.toBlock) and (not self.blockHash.is_empty):
            raise InvalidParamError(
                error_list="cannot specify both BlockHash and FromBlock/ToBlock, choose one or the other"
            )


class NpTxLogsApi(NeonProxyApi):
    name: ClassVar[str] = "NeonRPC::TransactionLogs"

    @NeonProxyApi.method(name="eth_getLogs")
    async def get_eth_logs(self, params: _RpcLogListRequest) -> list[RpcEthTxEventModel]:
        is_empty, from_block, to_block = await self._get_slot_range(params)
        if is_empty:
            return list()

        event_list = await self._db.get_event_list(from_block, to_block, params.address_list, params.topic_list)
        return [RpcEthTxEventModel.from_raw(e) for e in event_list if not e.is_hidden]

    @NeonProxyApi.method(name="neon_getLogs")
    async def get_neon_logs(self, params: _RpcLogListRequest) -> list[RpcNeonTxEventModel]:
        is_empty, from_block, to_block = await self._get_slot_range(params)
        if is_empty:
            return list()

        event_list = await self._db.get_event_list(from_block, to_block, params.address_list, params.topic_list)
        return [RpcNeonTxEventModel.from_raw(e) for e in event_list]

    async def _get_slot_range(self, params: _RpcLogListRequest) -> tuple[bool, int | None, int | None]:
        if not params.blockHash.is_empty:
            block = await self._db.get_block_by_hash(params.blockHash)
            return block.is_empty, block.slot, block.slot

        from_slot, to_slot = await asyncio.gather(self._get_slot(params.from_block), self._get_slot(params.to_block))
        if (from_slot is not None) and (to_slot is not None):
            if from_slot > to_slot:
                raise InvalidParamError(error_list="fromBlock can't be bigger than toBlock")
            # elif to_slot - from_slot > 1000:
            #     raise InvalidParamError(error_list="range is too large, max is 1000 blocks")
        return False, from_slot, to_slot

    async def _get_slot(self, block_tag: RpcBlockRequest | None) -> int | None:
        if not block_tag:
            return None

        block = await self.get_block_by_tag(block_tag)
        if block.is_empty:
            return None
        return block.slot
