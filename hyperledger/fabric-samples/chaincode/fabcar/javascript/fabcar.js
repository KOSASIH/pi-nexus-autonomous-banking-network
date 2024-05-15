'use strict';

const { Contract } = require('fabric-contract-api');

class FabcarContract extends Contract {

    async InitLedger(ctx) {
        console.info('Initializing the ledger');
        const cars = [
            {
                colour: 'red',
                make: 'Toyota',
                model: 'Corolla',
                owner: 'Tom'
            },
            {
              colour: 'blue',
                make: 'Ford',
                model: 'Mustang',
                owner: 'Kate'
            }
        ];

        for (let i = 0; i < cars.length; i++) {
            await this.CreateCar(ctx, cars[i].colour, cars[i].make, cars[i].model, cars[i].owner);
        }
        console.info('============= END : Initialize Ledger ===========');
    }

    async Init(ctx) {
        console.info('Initializing smart contract');
        await ctx.stub.putState('Init', Buffer.from('Init'));
        console.info('============= END : Initialize Smart Contract ===========');
    }

    async Invoke(ctx) {
        console.info('Executing transaction ...');
        const functionName = ctx.stub.getFunctionAndParameters()[0];
        console.info('Function: ' + functionName);
        let ret = undefined;
        switch (functionName) {
            case 'CreateCar':
                ret = await this.CreateCar(ctx, ctx.stub.getArgs()[0], ctx.stub.getArgs()[1], ctx.stub.getArgs()[2], ctx.stub.getArgs()[3]);
                break;
            case 'QueryCar':
                ret = await this.QueryCar(ctx, ctx.stub.getArgs()[0]);
                break;
            case 'ChangeCarOwner':
                ret = await this.ChangeCarOwner(ctx, ctx.stub.getArgs()[0], ctx.stub.getArgs()[1]);
                break;
            default:
                throw new Error('Received unknown function ' + functionName + ' invocation');
        }
        console.info('============= END : transaction execution ===========');
        return ret;
    }

    async QueryCar(ctx, carNumber) {
        const carAsBytes = await ctx.stub.getState(carNumber);
        if (!carAsBytes || carAsBytes.length === 0) {
            throw new Error('Car not found: ' + carNumber);
        }
        console.log(carAsBytes.toString());
        return carAsBytes.toString();
    }

    async CreateCar(ctx, colour, make, model, owner) {
        console.info('============= START : Create Car ===========');
        const car = {
            colour,
            make,
            model,
            owner
        };
        const carAsBytes = Buffer.from(JSON.stringify(car));
        await ctx.stub.putState(colour, carAsBytes);
        console.info('============= END : Create Car ===========');
    }

    async ChangeCarOwner(ctx, carNumber, newOwner) {
        console.info('============= START : Change Car Owner ===========');
        const carAsBytes = await ctx.stub.getState(carNumber);
        if (!carAsBytes || carAsBytes.length === 0) {
            throw new Error('Car not found: ' + carNumber);
        }
        const car = JSON.parse(carAsBytes.toString());
        car.owner = newOwner;
        const carAsBytesUpdated = Buffer.from(JSON.stringify(car));
        await ctx.stub.putState(carNumber, carAsBytesUpdated);
        console.info('============= END : Change Car Owner ===========');
    }

}

module.exports = FabcarContract;
