import React from "react";
import { render, fireEvent, waitFor } from "@testing-library/react";
import { PropertyListing } from "../PropertyListing";
import { MockedProvider } from "@apollo/client/testing";
import { GET_PROPERTIES } from "../graphql/queries";
import { CREATE_PROPERTY } from "../graphql/mutations";

describe("PropertyListing component", () => {
  const properties = [
    {
      id: "1",
      address: "123 Main St",
      city: "Anytown",
      state: "CA",
      zip: "12345",
      price: 100000,
    },
    {
      id: "2",
      address: "456 Elm St",
      city: "Othertown",
      state: "CA",
      zip: "67890",
      price: 200000,
    },
  ];

  const mocks = [
    {
      request: {
        query: GET_PROPERTIES,
      },
      result: {
        data: {
          properties,
        },
      },
    },
    {
      request: {
        query: CREATE_PROPERTY,
        variables: {
          address: "789 Oak St",
          city: "Thistown",
          state: "CA",
          zip: "12345",
          price: 250000,
        },
      },
      result: {
        data: {
          createProperty: {
            id: "3",
            address: "789 Oak St",
            city: "Thistown",
            state: "CA",
            zip: "12345",
            price: 250000,
          },
        },
      },
    },
  ];

  it("renders a list of properties", async () => {
    const { getByText } = render(
      <MockedProvider mocks={mocks}>
        <PropertyListing />
      </MockedProvider>
    );

    await waitFor(() => getByText("123 Main St"));
    expect(getByText("456 Elm St")).toBeInTheDocument();
  });

  it("allows user to create a new property", async () => {
    const { getByText, getByLabelText } = render(
      <MockedProvider mocks={mocks}>
        <PropertyListing />
      </MockedProvider>
    );

    const addressInput = getByLabelText("Address");
    const cityInput = getByLabelText("City");
    const stateInput = getByLabelText("State");
    const zipInput = getByLabelText("Zip");
    const priceInput = getByLabelText("Price");
    const createButton = getByText("Create Property");

    fireEvent.change(addressInput, { target: { value: "789 Oak St" } });
    fireEvent.change(cityInput, { target: { value: "Thistown" } });
    fireEvent.change(stateInput, { target: { value: "CA" } });
    fireEvent.change(zipInput, { target: { value: "12345" } });
    fireEvent.change(priceInput, { target: { value: 250000 } });

    fireEvent.click(createButton);

    await waitFor(() => getByText("789 Oak St"));
    expect(getByText("Thistown")).toBeInTheDocument();
  });
});
