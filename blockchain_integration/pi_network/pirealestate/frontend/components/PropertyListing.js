import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { PropertyService } from "../services/PropertyService";
import { Spinner } from "./Spinner";
import { ErrorComponent } from "./ErrorComponent";

const PropertyListing = () => {
  const [properties, setProperties] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchProperties = async () => {
      try {
        setLoading(true);
        const response = await PropertyService.getProperties();
        setProperties(response.data);
        setLoading(false);
      } catch (error) {
        setError(error);
        setLoading(false);
      }
    };
    fetchProperties();
  }, []);

  if (loading) {
    return <Spinner />;
  }

  if (error) {
    return <ErrorComponent error={error} />;
  }

  return (
    <div className="property-listing">
      <h1>Property Listing</h1>
      <ul>
        {properties.map((property) => (
          <li key={property._id}>
            <Link to={`/properties/${property._id}`}>
              <img src={property.image} alt={property.name} />
              <h2>{property.name}</h2>
              <p>{property.description}</p>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default PropertyListing;
