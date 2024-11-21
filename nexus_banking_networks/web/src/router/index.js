import Vue from "vue";
import VueRouter from "vue-router";
import AccountsView from "../views/AccountsView.vue";

Vue.use(VueRouter);

const routes = [
  {
    path: "/",
    name: "Accounts",
    component: AccountsView,
  },
];

const router = new VueRouter({
  mode: "history",
  base: process.env.BASE_URL,
  routes,
});

export default router;
