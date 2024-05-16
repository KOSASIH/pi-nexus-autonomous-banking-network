async function initLedger(ctx) {
  console.info('Initializing Ledger');
  const coins = [
    {
      owner: 'Tomoko',
      amount: '1000000',
    },
    {
      owner: 'Brad',
      amount: '2000000',
    },
    {
      owner: 'Jin Soo',
      amount: '3000000',
    },
    {
      owner: 'Max',
      amount: '4000000',
    },
    {
      owner: 'Adriana',
      amount: '5000000',
    },
  ];

  for (let i = 0; i < coins.length; i++) {
    coins[i].docType = 'coin';
    await ctx.stub.putState(`${i}`, Buffer.from(JSON.stringify(coins[i])));
    console.info(`Added <--> ${coins[i].owner} ${coins[i].amount} coins`);
  }
  console.info('Initialization complete');
}
