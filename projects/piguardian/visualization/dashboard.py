import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

class Dashboard:
    def __init__(self):
        self.df = pd.read_csv('data/us-population-2010-2019-reshaped.csv')

    def create_dashboard(self):
        st.set_page_config(
            page_title="US Population Dashboard",
            page_icon="🏂",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        alt.themes.enable("dark")

        with st.sidebar:
            st.title('🏂 US Population Dashboard')

            year_list = list(self.df.year.unique())[::-1]
            selected_year = st.selectbox('Select a year', year_list, index=len(year_list)-1)
            df_selected_year = self.df[self.df.year == selected_year]
            df_selected_year_sorted = df_selected_year.sort_values(by="population", ascending=False)

            color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
            selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

        st.header("US Population Dashboard")
        st.write("This dashboard displays the population of the 52 US states from 2010 to 2019.")

        self.create_choropleth_map(df_selected_year_sorted, selected_color_theme)
        self.create_heatmap(df_selected_year_sorted, selected_color_theme)
        self.create_donut_chart(df_selected_year_sorted)
        self.create_metrics_card(df_selected_year_sorted)

    def create_choropleth_map(self, df, color_theme):
        choropleth = alt.Chart(df).mark_geoshape().encode(
            color=alt.Color('population:Q', scale=alt.Scale(scheme=color_theme)),
            tooltip=['state:N', 'population:Q']
        ).properties(
            width=800,
            height=500
        )
        st.altair_chart(choropleth)

    def create_heatmap(self, df, color_theme):
        heatmap = alt.Chart(df).mark_rect().encode(
            x='state:N',
            y='year:O',
            color=alt.Color('population:Q', scale=alt.Scale(scheme=color_theme))
        ).properties(
            width=800,
            height=500
        )
        st.altair_chart(heatmap)

    def create_donut_chart(self, df):
        donut_chart = px.pie(df, names='state', values='population')
        st.plotly_chart(donut_chart)

    def create_metrics_card(self, df):
        st.metric(label='Total Population', value=df['population'].sum())
        st.metric(label='Average Population', value=df['population'].mean())

if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.create_dashboard()
