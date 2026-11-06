"""Model economics with the official shop reader, navigation and enhancement."""
import uuid
from custom.action.climb_tower_shop.interactor import ShopInteractor
from custom.reco.climb_tower_potential.state import State

from .tower_sampling_collector import ObservedShopHandler, snapshot
from .value_model import Action, sim, replace


class ShopNavigationFailed(RuntimeError):
    """A model shop action cannot continue on the current page."""


class ModelShopInteractor(ShopInteractor):
    # The upstream handler calls these internally and ignores False.
    def enter_shopping(self):
        if not super().enter_shopping():
            raise ShopNavigationFailed("enter_shopping")
        return True

    def back_to_main_page(self):
        if not super().back_to_main_page():
            raise ShopNavigationFailed("back_to_main_page")
        return True


def run_model_shop(action_handler, context, argv, runtime):
    tracker = runtime.require_tracker()
    data = action_handler._init_data(context, argv.node_name)
    interactor = ModelShopInteractor(context)
    handler = ObservedShopHandler(interactor, data, runtime.collector)
    # Read only the fields used by the model; notes are outside its actions.
    handler._update_enhancement_cost()
    handler._update_shop_type()
    handler.read_goods_page_info()
    tracker.sync_owned(State.owned_potentials)
    tracker.observe_route("SHOP_UNSEEN", data.current_coin)
    tracker.state = replace(tracker.state, phase="SHOP")
    while not context.tasker.stopping:
        state = tracker.state
        cost = data.current_enhancement_cost
        if cost < 0:
            raise ValueError("神经网络未读到商店强化费用")
        paid_index = sim.PRICES.index(cost) if cost else state.paidIndex
        stock = [[0, 0, 0], [0, 0, 0]]
        goods, actions = {}, []
        for item in data.items:
            if item.internal_name != "potential_drink" or item.bought:
                continue
            drink_type = 2 if item.trekker_specified else 1
            index = sim.DRINK_PRICES.index(item.price)+1
            stock[drink_type-1][index-1] += 1
            if item.price <= data.current_coin:
                action = Action("BUY", index=index, price=item.price, type=drink_type)
                if action not in goods:
                    actions.append(action)
                    goods[action] = item
        if cost <= data.current_coin and any(row[3] < 6 for row in state.H):
            actions.append(Action("ENHANCE", price=cost))
        if data.refresh_remaining > 0 and data.refresh_cost <= data.current_coin:
            actions.append(Action("RESTOCK", price=data.refresh_cost))
        actions.append(Action("LEAVE"))
        tracker.state = replace(state, phase="SHOP", coins=data.current_coin,
                                freeEnhance=cost == 0, paidIndex=paid_index,
                                stock=tuple(tuple(row) for row in stock), restocks=data.refresh_remaining,
                                source=None, offer=())
        # Match the critic's action catalogue, including the frozen
        # equivalence grouping of identical drinks at different prices.
        available = set(actions)
        actions = [a for a in sim.legal_actions(tracker.state) if a in available]
        action = tracker.choose(tuple(actions))
        if action.kind == "LEAVE":
            interactor.back_to_main_page()
            tracker.state = replace(tracker.state, phase=tracker.state.shopReturnPhase,
                                    source=None, stock=sim.EMPTY_STOCK,
                                    paidIndex=0, firstInitial=False)
            return True
        if action.kind == "RESTOCK":
            handler.refresh()
            if context.tasker.stopping:
                return False
            handler.read_goods_page_info()
            tracker.state = replace(tracker.state, firstInitial=False)
            continue
        if action.kind == "ENHANCE":
            before_enhance = tracker.state
            tracker.state = replace(before_enhance, freeEnhance=False,
                                    paidIndex=paid_index if cost == 0 else min(6, paid_index+1),
                                    source=sim.Source("regular", "any", 0, "SHOP", False))
            try:
                enhanced = handler._enhance()
            except ShopNavigationFailed:
                tracker.state = before_enhance
                raise
            if not enhanced:
                return False
            interactor.enter_shopping()
            handler.update_coin()
            continue
        item = goods[action]
        source = "specified_drink" if item.trekker_specified else "drink"
        transaction = dict(operation_id=uuid.uuid4().hex, stock_scan_id=runtime.collector.stock_scan_id,
                           item=snapshot(item), source_type=source)
        before_coin = data.current_coin
        runtime.collector.emit("purchase_attempt", phase="before", coins=before_coin,
                               owned=runtime.collector.owned(), **transaction)
        with runtime.collector.action_scope(**transaction), runtime.collector.capture("purchase", **transaction):
            result = context.run_task("星塔_节点_商店_购物_购买道具_agent", {
                "星塔_节点_商店_购物_购买道具_agent": {"action": {"param": {"target": item.price_roi}}},
                "星塔_节点_选择潜能_agent": {"attach": {"reserved_coin": 0, "potential_source": source}}
            })
        if not (result and result.status.succeeded):
            runtime.collector.emit("purchase_result", success=False, pipeline_succeeded=False,
                                   debit_confirmed=None, **transaction)
            return False
        item.bought = True
        interactor.update_image()
        handler.update_coin()
        debit_confirmed = True if 0 <= data.current_coin <= before_coin - item.price else None
        runtime.collector.emit("purchase_result", phase="after", success=True,
            pipeline_succeeded=True, debit_confirmed=debit_confirmed,
            coins_before=before_coin, coins_after=data.current_coin, owned=runtime.collector.owned(),
            confirmation_basis="Maa_purchase_pipeline_return", **transaction)
    return False

