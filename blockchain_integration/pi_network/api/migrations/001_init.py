from alembic import op

revision = "001"
down_revision = None

def upgrade():
    op.create_table("users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password", sa.String(length=128), nullable=False),
        sa.Column("email", sa.String(length=120), nullable=False)
    )

def downgrade():
    op.drop_table("users")
