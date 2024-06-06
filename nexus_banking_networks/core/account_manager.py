import os
import sys
import logging
from flask_sqlalchemy import SQLAlchemy

logger = logging.getLogger(__name__)

class AccountManager:
    def __init__(self):
        self.db = SQLAlchemy()

    def create_account(self, data):
        # Create account using SQLAlchemy
        account = Account(**data)
        self.db.session.add(account)
        self.db.session.commit()
        logger.info('Account created successfully')
        return {'message': 'Account created successfully'}, 201

    def update_account(self, data):
        # Update account using SQLAlchemy
        account = self.db.session.query(Account).filter_by(id=data['id']).first()
        if account:
            account.update(**data)
            self.db.session.commit()
            logger.info('Account updated successfully')
            return {'message': 'Account updated successfully'}, 200
        else:
            return {'error': 'Account not found'}, 404

    def delete_account(self, data):
        # Delete account using SQLAlchemy
        account = self.db.session.query(Account).filter_by(id=data['id']).first()
        if account:
            self.db.session.delete(account)
            self.db.session.commit()
            logger.info('Account deleted successfully')
            return {'message': 'Account deleted successfully'}, 200
        else:
            return {'error': 'Account not found'}, 404
