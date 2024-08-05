import React from 'eact';
import { Link } from 'eact-router-dom';
import { Container, Row, Col } from 'eactstrap';

const Header = () => {
  return (
    <header>
      <Container>
        <Row>
          <Col xs="12" sm="6" md="4" lg="3">
            <Link to="/" className="logo">
              <img src="logo.png" alt="Logo" />
            </Link>
          </Col>
          <Col xs="12" sm="6" md="8" lg="9">
            <nav>
              <ul>
                <li>
                  <Link to="/about">About</Link>
                </li>
                <li>
                  <Link to="/contact">Contact</Link>
                </li>
              </ul>
            </nav>
          </Col>
        </Row>
      </Container>
    </header>
  );
};

export default Header;
