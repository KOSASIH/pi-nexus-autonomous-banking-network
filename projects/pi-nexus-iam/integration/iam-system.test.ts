import { Chance } from "chance";
import { Context, PostConfirmationConfirmSignUpTriggerEvent } from "aws-lambda";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, GetCommand } from "@aws-sdk/lib-dynamodb";
import { handler as iamSystem } from "../../functions/iam-system";

const MOCKED_CONTEXT: Context = {
  callbackWaitsForEmptyEventLoop: false,
  functionName: "mocked",
  functionVersion: "mocked",
  invokedFunctionArn: "mocked",
  memoryLimitInMB: "mocked",
  awsRequestId: "mocked",
  logGroupName: "mocked",
  logStreamName: "mocked",
  getRemainingTimeInMillis(): number {
    return 999;
  },
  done(error?: Error, result?: any): void {
    return;
  },
  fail(error: Error | string): void {
    return;
  },
  succeed(messageOrObject: any): void {
    return;
  },
};

describe("When iamSystem runs", () => {
  it("The user's profile should be saved in DynamoDb", async () => {
    // Arrange
    const firstName = Chance().first({ nationality: "en" });
    const lastName = Chance().last({ nationality: "en" });
    const name = `${firstName} ${lastName}`;
    const email = `${firstName}-${lastName}@test.com`;
    const username = Chance().guid();

    // Act
    const event: PostConfirmationConfirmSignUpTriggerEvent = {
      version: "1",
      region: process.env.AWS_REGION!,
      userPoolId: process.env.COGNITO_USER_POOL_ID!,
      userName: username,
      triggerSource: "PostConfirmation_ConfirmSignUp",
      request: {
        userAttributes: {
          sub: username,
          "cognito:email_alias": email,
          "cognito:user_status": "CONFIRMED",
          email_verified: "false",
          name: name,
          email: email,
        },
      },
      response: {},
      callerContext: {
        awsSdkVersion: "3",
        clientId: "string",
      },
    };
    await iamSystem(event, MOCKED_CONTEXT, () => {});

    // Assert
    const client = new DynamoDBClient({});
    const ddDocClient = DynamoDBDocumentClient.from(client);

    const getUser = new GetCommand({
      TableName: process.env.USERS_TABLE,
      Key: {
        id: username,
      },
    });
    const resp = await ddDocClient.send(getUser);
    const ddbUser = resp.Item;

    expect(ddbUser).toMatchObject({
      id: username,
      name,
      createdAt: expect.stringMatching(
        /\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d(?:\.\d+)?Z?/g
      ),
    });
  });
});
