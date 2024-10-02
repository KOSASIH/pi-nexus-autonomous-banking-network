import React from "eact";
import { render, fireEvent, waitFor } from "@testing-library/react";
import { MyContract } from "./MyContract";

describe("MyContract", () => {
  it("should render correctly", () => {
    const { getByText } = render(<MyContract />);
    expect(getByText("My Contract")).toBeInTheDocument();
  });

  it("should call the contract function correctly", async () => {
    const contractFunction = jest.fn();
    const { getByText } = render(
      <MyContract contractFunction={contractFunction} />,
    );
    const button = getByText("Call Contract Function");
    fireEvent.click(button);
    await waitFor(() => expect(contractFunction).toHaveBeenCalledTimes(1));
  });

  it("should handle errors correctly", async () => {
    const contractFunction = jest.fn(() => {
      throw new Error("Contract function error");
    });
    const { getByText } = render(
      <MyContract contractFunction={contractFunction} />,
    );
    const button = getByText("Call Contract Function");
    fireEvent.click(button);
    await waitFor(() =>
      expect(getByText("Error: Contract function error")).toBeInTheDocument(),
    );
  });
});
