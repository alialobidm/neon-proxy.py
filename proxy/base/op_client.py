from __future__ import annotations

from typing import Sequence

from common.app_data.client import AppDataClient
from common.solana.pubkey import SolPubKey
from common.solana.transaction import SolTx
from common.solana.transaction_model import SolTxModel
from .op_api import (
    OP_RESOURCE_ENDPOINT,
    OpResourceModel,
    OpGetResourceRequest,
    OpFreeResourceRequest,
    OpResourceResp,
    OpTokenSolAddressModel,
    OpGetTokenSolAddressRequest,
    OpSignSolTxListRequest,
    OpSolTxListResp,
    OpGetSignerKeyListRequest,
    OpSignerKeyListResp,
)


class OpResourceClient(AppDataClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.connect(host="127.0.0.1", port=self._cfg.op_resource_port, path=OP_RESOURCE_ENDPOINT)

    async def get_resource(self, req_id: dict, chain_id: int | None) -> OpResourceModel:
        return await self._get_resource(OpGetResourceRequest(req_id=req_id, chain_id=chain_id))

    async def free_resource(self, req_id: dict, is_good_resource: bool, resource: OpResourceModel) -> bool:
        req = OpFreeResourceRequest(req_id=req_id, is_good=is_good_resource, resource=resource)
        resp = await self._free_resource(req)
        return resp.result

    async def get_token_sol_address(self, req_id: dict, owner: SolPubKey, chain_id: int) -> SolPubKey:
        req = OpGetTokenSolAddressRequest(req_id=req_id, owner=owner, chain_id=chain_id)
        resp = await self._get_token_sol_address(req)
        return resp.token_sol_address

    async def sign_sol_tx_list(self, req_id: dict, owner: SolPubKey, tx_list: Sequence[SolTx]) -> tuple[SolTx, ...]:
        model_list = [SolTxModel.from_raw(tx) for tx in tx_list]
        req = OpSignSolTxListRequest(req_id=req_id, owner=owner, tx_list=model_list)
        resp = await self._sign_sol_tx_list(req)
        return tuple([model.tx for model in resp.tx_list])

    async def get_signer_key_list(self, req_id: dict) -> tuple[SolPubKey, ...]:
        req = OpGetSignerKeyListRequest(req_id=req_id)
        resp = await self._get_signer_key_list(req)
        return tuple(resp.signer_key_list)

    @AppDataClient.method(name="getOperatorResource")
    async def _get_resource(self, request: OpGetResourceRequest) -> OpResourceModel: ...

    @AppDataClient.method(name="freeOperatorResource")
    async def _free_resource(self, request: OpFreeResourceRequest) -> OpResourceResp: ...

    @AppDataClient.method(name="getOperatorTokenAddress")
    async def _get_token_sol_address(self, request: OpGetTokenSolAddressRequest) -> OpTokenSolAddressModel: ...

    @AppDataClient.method(name="signSolanaTransactionList")
    async def _sign_sol_tx_list(self, request: OpSignSolTxListRequest) -> OpSolTxListResp: ...

    @AppDataClient.method(name="getSignerKeyList")
    async def _get_signer_key_list(self, request: OpGetSignerKeyListRequest) -> OpSignerKeyListResp: ...
