<template>
  <v-card>
    <v-card-title>
      <h3 class="headline">{{ account.account_id }}</h3>
    </v-card-title>
    <v-card-text>
      <p>Balance: {{ account.balance }}</p>
    </v-card-text>
  </v-card>
</template>

<script>
export default {
  props: {
    account: {
      type: Object,
      required: true
    }
  }
}
</script>
