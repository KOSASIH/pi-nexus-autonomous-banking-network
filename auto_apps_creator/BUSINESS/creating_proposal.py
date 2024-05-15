import os
import json
import datetime
from jinja2 import Template, Environment, FileSystemLoader
from pdfkit import from_string
from PIL import Image
from flask import Flask, request, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key_here'

class BusinessProposalForm(FlaskForm):
    company_name = StringField('Company Name', validators=[DataRequired()])
    proposal_title = StringField('Proposal Title', validators=[DataRequired()])
    proposal_description = TextAreaField('Proposal Description', validators=[DataRequired()])
    target_market = StringField('Target Market', validators=[DataRequired()])
    competitive_advantage = StringField('Competitive Advantage', validators=[DataRequired()])
    financial_projections = StringField('Financial Projections', validators=[DataRequired()])
    team_members = StringField('Team Members', validators=[DataRequired()])
    technology_stack = StringField('Technology Stack', validators=[DataRequired()])

@app.route('/', methods=['GET', 'POST'])
def index():
    form = BusinessProposalForm()
    if form.validate_on_submit():
        company_name = form.company_name.data
        proposal_title = form.proposal_title.data
        proposal_description = form.proposal_description.data
        target_market = form.target_market.data
        competitive_advantage = form.competitive_advantage.data
        financial_projections = form.financial_projections.data
        team_members = form.team_members.data
        technology_stack = form.technology_stack.data

        # Load the proposal template
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('proposal_template.html')

        # Prepare the data for the template
        data = {
            'company_name': company_name,
            'proposal_title': proposal_title,
            'proposal_description': proposal_description,
            'target_market': target_market,
            'competitive_advantage': competitive_advantage,
            'financial_projections': financial_projections,
            'team_members': team_members,
            'technology_stack': technology_stack,
            'date': datetime.date.today().strftime('%B %d, %Y')
        }

        # Render the template with the data
        proposal = template.render(data)

        # Generate a PDF report
        pdf = from_string(proposal, False)

        # Save the PDF report
        with open('proposal.pdf', 'wb') as f:
            f.write(pdf)

        # Return a success message
        return 'Proposal generated successfully!'

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
