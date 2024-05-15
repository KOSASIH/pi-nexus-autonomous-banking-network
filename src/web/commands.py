from flask import current_app
from flask_script import Manager, Command
from flask_migrate import MigrateCommand

from src.web.extensions import db
from src.web.models import User

manager = Manager(use_default_colors=True)
migrate = MigrateCommand()

manager.add_command('db', migrate)

@manager.command
def seed():
    """Seed the database with sample data."""
    db.session.add(User(username='admin', email='admin@example.com', password='password'))
    db.session.commit()

@manager.command
def worker():
    """Run a background worker process."""
    from src.web.tasks import process_images

    while True:
        process_images.delay()
        current_app.logger.info('Processed an image.')

if __name__ == '__main__':
    manager.run()
