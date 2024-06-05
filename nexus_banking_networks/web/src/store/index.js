import Vue from "vue";
import Vuex from "vuex";
import account from "./modules/account";
import transaction from "./modules/transaction";

Vue.use(Vuex);

export default new Vuex.Store({
  modules: {
    account,
    transaction,
  },
});
