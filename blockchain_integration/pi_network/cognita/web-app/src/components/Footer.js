import React from 'eact';
import { Container, Row, Col } from 'eactstrap';

const Footer = () => {
  return (
    <footer>
      <Container>
        <Row>
          <Col xs="12" sm="6" md="4" lg="3">
            <h5>About Us</h5>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
          </Col>
          <Col xs="12" sm="6" md="4" lg="3">
            <h5>Quick Links</h5>
            <ul>
              <li>
                <a href="#">Link 1</a>
              </li>
              <li>
                <a href="#">Link 2</a>
              </li>
              <li>
                <a href="#">Link 3</a>
              </li>
            </ul>
          </Col>
          <Col xs="12" sm="6" md="4" lg="3">
            <h5>Follow Us</h5>
            <ul>
              <li>
                <a href="#" target="_blank">
                  <i className="fab fa-facebook-f" />
                </a>
              </li>
              <li>
                <a href="#" target="_blank">
                  <i className="fab fa-twitter" />
                </a>
              </li>
              <li>
                <a href="#" target="_blank">
                  <i className="fab fa-instagram" />
                </a>
              </li>
            </ul>
          </Col>
        </Row>
        <Row>
          <Col xs="12" className="text-center">
            <p>&copy; 2023 AI Platform. All rights reserved.</p>
          </Col>
        </Row>
      </Container>
    </footer>
  );
};

export default Footer;
