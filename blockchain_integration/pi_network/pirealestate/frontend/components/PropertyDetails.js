import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { PropertyService } from "../services/PropertyService";
import { Spinner } from "./Spinner";
import { ErrorComponent } from "./ErrorComponent";
import { MapComponent } from "./MapComponent";

const PropertyDetails = () => {
  const { id } = useParams();
  const [property, setProperty] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchProperty = async () => {
      try {
        setLoading(true);
        const response = await PropertyService.getProperty(id);
        setProperty(response.data);
        setLoading(false);
      } catch (error) {
        setError(error);
        setLoading(false);
      }
    };
    fetchProperty();
  }, [id]);

  if (loading) {
    return <Spinner />;
  }

  if (error) {
    return <ErrorComponent error={error} />;
  }

  return (
    <div className="property-details">
      <h1>{property.name}</h1>
      <img src={property.image} alt={property.name} />
      <p>{property.description}</p>
      <MapComponent latitude={property.latitude} longitude={property.longitude} />
      <ul>
        <li>
          <strong>Price:</strong> ${property.price}
        </li>
        <li>
          <strong>Location:</strong> {property.location}
        </li>
        <li>
          <strong>Owner:</strong> {property.owner.name}
        </li>
      </ul>
    </div>
  );
};

export default PropertyDetails;
