import { expect } from "chai";
import { supertest } from "supertest";
import app from "../app";
import { Property } from "../models/Property";
import { User } from "../models/User";
import { authenticate } from "../utils/authenticate";
import { createProperty } from "../utils/createProperty";
import { createTestUser } from "../utils/createTestUser";

describe("Property API", () => {
  let request;
  let user;
  let property;

  beforeEach(async () => {
    request = supertest(app);
    user = await createTestUser();
    property = await createProperty(user);
  });

  describe("GET /properties", () => {
    it("should return a list of properties", async () => {
      const res = await request.get("/properties");
      expect(res.status).to.equal(200);
      expect(res.body).to.be.an("array");
      expect(res.body.length).to.be.above(0);
    });

    it("should return a list of properties filtered by city", async () => {
      const res = await request.get("/properties?city=Anytown");
      expect(res.status).to.equal(200);
      expect(res.body).to.be.an("array");
      expect(res.body.every((property) => property.city === "Anytown")).to.be.true;
    });
  });

  describe("GET /properties/:id", () => {
    it("should return a single property", async () => {
      const res = await request.get(`/properties/${property.id}`);
      expect(res.status).to.equal(200);
      expect(res.body).to.be.an("object");
      expect(res.body.id).to.equal(property.id);
    });

    it("should return a 404 error for a non-existent property", async () => {
      const res = await request.get("/properties/1234567890");
      expect(res.status).to.equal(404);
    });
  });

  describe("POST /properties", () => {
    it("should create a new property", async () => {
      const newPropertyData = {
        address: "456 Elm St",
        city: "Othertown",
        state: "CA",
        zip: "12345",
        price: 200000,
      };
      const res = await request.post("/properties").send(newPropertyData);
      expect(res.status).to.equal(201);
      expect(res.body).to.be.an("object");
      expect(res.body.address).to.equal(newPropertyData.address);
    });

    it("should return a 401 error for an unauthorized user", async () => {
      const newPropertyData = {
        address: "456 Elm St",
        city: "Othertown",
        state: "CA",
        zip: "12345",
        price: 200000,
      };
      const res = await request.post("/properties").send(newPropertyData);
      expect(res.status).to.equal(401);
    });
  });

  describe("PUT /properties/:id", () => {
    it("should update an existing property", async () => {
      const updatedPropertyData = {
        address: "789 Oak St",
        city: "Thistown",
        state: "CA",
        zip: "12345",
        price: 250000,
      };
      const res = await request.put(`/properties/${property.id}`).send(updatedPropertyData);
      expect(res.status).to.equal(200);
      expect(res.body).to.be.an("object");
      expect(res.body.address).to.equal(updatedPropertyData.address);
    });

    it("should return a 404 error for a non-existent property", async () => {
      const updatedPropertyData = {
        address: "789 Oak St",
        city: "Thistown",
        state: "CA",
        zip: "12345",
        price: 250000,
      };
      const res = await request.put("/properties/1234567890").send(updatedPropertyData);
      expect(res.status).to.equal(404);
    });
  });

  describe("DELETE /properties/:id", () => {
    it("should delete an existing property", async () => {
      const res = await request.delete(`/properties/${property.id}`);
      expect(res.status).to.equal(204);
    });

    it("should return a 404 error for a non-existent property", async () => {
      const res = await request.delete("/properties/1234567890");
      expect(res.status).to.equal(404);
    });
  });
});
