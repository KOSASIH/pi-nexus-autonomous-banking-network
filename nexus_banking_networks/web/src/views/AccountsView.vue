<template>
  <v-container>
    <v-row>
      <v-col
        v-for="account in accounts"
        :key="account.account_id"
        cols="12"
        sm="6"
        md="4"
      >
        <AccountCard :account="account" />
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import AccountCard from '../components/AccountCard.vue'

export default {
  components: {
    AccountCard
  },
  computed: {
    accounts() {
      return this.$store.state.account.accounts
    }
  }
}
</script>
