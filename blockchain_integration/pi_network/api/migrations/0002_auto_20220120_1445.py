# 0002_auto_20220120_1445.py
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='pitransaction',
            unique_together={('tx_id', 'sender', 'recipient')},
        ),
    ]
