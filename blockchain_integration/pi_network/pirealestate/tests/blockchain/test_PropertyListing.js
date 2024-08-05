import { expect } from "chai";
import { ethers } from "ethers";
import { PropertyListing } from "../contracts/PropertyListing";
import { deployContract } from "../utils/deployContract";
import { getTestProvider } from "../utils/getTestProvider";

describe("PropertyListing contract", () => {
  let provider;
  let contract;
  let owner;
  let nonOwner;

  beforeEach(async () => {
    provider = await getTestProvider();
    [owner, nonOwner] = await provider.listAccounts();
    contract = await deployContract("PropertyListing", [owner]);
  });

  describe("listProperty", () => {
    it("should allow owner to list a property", async () => {
      const propertyData = {
        address: "123 Main St",
        city: "Anytown",
        state: "CA",
        zip: "12345",
        price: ethers.utils.parseEther("100000"),
      };
      await contract.listProperty(propertyData);
      const listedProperties = await contract.getListedProperties();
      expect(listedProperties.length).to.equal(1);
      expect(listedProperties[0].address).to.equal(propertyData.address);
    });

    it("should not allow non-owner to list a property", async () => {
      const propertyData = {
        address: "123 Main St",
        city: "Anytown",
        state: "CA",
        zip: "12345",
        price: ethers.utils.parseEther("100000"),
      };
      await expect(contract.connect(nonOwner).listProperty(propertyData)).to.be.revertedWith(
        "Only the owner can list a property"
      );
    });
  });

  describe("updateProperty", () => {
    it("should allow owner to update a property", async () => {
      const propertyData = {
        address: "123 Main St",
        city: "Anytown",
        state: "CA",
        zip: "12345",
        price: ethers.utils.parseEther("100000"),
      };
      await contract.listProperty(propertyData);
      const updatedPropertyData = {
        ...propertyData,
        price: ethers.utils.parseEther("120000"),
      };
      await contract.updateProperty(updatedPropertyData);
      const listedProperties = await contract.getListedProperties();
      expect(listedProperties[0].price).to.equal(updatedPropertyData.price);
    });

    it("should not allow non-owner to update a property", async () => {
      const propertyData = {
        address: "123 Main St",
        city: "Anytown",
        state: "CA",
        zip: "12345",
        price: ethers.utils.parseEther("100000"),
      };
      await contract.listProperty(propertyData);
      const updatedPropertyData = {
        ...propertyData,
        price: ethers.utils.parseEther("120000"),
      };
      await expect(contract.connect(nonOwner).updateProperty(updatedPropertyData)).to.be.revertedWith(
        "Only the owner can update a property"
      );
    });
  });

  describe("removeProperty", () => {
    it("should allow owner to remove a property", async () => {
      const propertyData = {
        address: "123 Main St",
        city: "Anytown",
        state: "CA",
        zip: "12345",
        price: ethers.utils.parseEther("100000"),
      };
      await contract.listProperty(propertyData);
      await contract.removeProperty(propertyData.address);
      const listedProperties = await contract.getListedProperties();
      expect(listedProperties.length).to.equal(0);
    });

    it("should not allow non-owner to remove a property", async () => {
      const propertyData = {
        address: "123 Main St",
        city: "Anytown",
        state: "CA",
        zip: "12345",
        price: ethers.utils.parseEther("100000"),
      };
      await contract.listProperty(propertyData);
      await expect(contract.connect(nonOwner).removeProperty(propertyData.address)).to.be.revertedWith(
        "Only the owner can remove a property"
      );
    });
  });
});
