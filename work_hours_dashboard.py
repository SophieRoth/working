import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROJECT_DIR = Path(__file__).resolve().parent

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": [
        "zoom2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
    ],
}


def decimal_to_time_string(decimal_hour: float) -> str:
    if pd.isna(decimal_hour):
        return "N/A"
    h = int(decimal_hour)
    m = int(round((decimal_hour - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"


def parse_datetime_column(values: pd.Series) -> pd.Series:
    def parse_one(value):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return pd.NaT
        if getattr(parsed, "tzinfo", None) is not None:
            return parsed.tz_localize(None)
        return parsed

    return values.apply(parse_one)


def clean_hours_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["start_clean"] = df["start"].astype(str).str.replace(r"\s*\(.*\)", "", regex=True)
    df["end_clean"] = df["end"].astype(str).str.replace(r"\s*\(.*\)", "", regex=True)
    df["start_dt"] = parse_datetime_column(df["start_clean"])
    df["end_dt"] = parse_datetime_column(df["end_clean"])
    df = df.dropna(subset=["start_dt", "end_dt"]).sort_values("start_dt")

    df["hours"] = df["hours"].astype(str).str.replace(",", ".").str.strip()
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
    return df.dropna(subset=["hours"]).copy()


def get_property(properties: dict, name: str) -> dict:
    try:
        return properties[name]
    except KeyError as exc:
        raise KeyError(f"Notion database is missing expected property: {name}") from exc


def notion_date_value(properties: dict, name: str) -> Optional[str]:
    prop = get_property(properties, name)
    if prop.get("type") != "date" or not prop.get("date"):
        return None
    return prop["date"].get("start")


def notion_number_value(properties: dict, name: str) -> Optional[float]:
    prop = get_property(properties, name)
    prop_type = prop.get("type")
    if prop_type == "number":
        return prop.get("number")
    if prop_type == "formula" and prop.get("formula", {}).get("type") == "number":
        return prop["formula"].get("number")
    return None


def load_data_from_notion() -> pd.DataFrame:
    token = os.environ.get("NOTION_TOKEN")
    database_id = os.environ.get("NOTION_DATABASE_ID")
    if not token or not database_id:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATABASE_ID must both be set.")

    start_property = os.environ.get("NOTION_START_PROPERTY") or "start"
    end_property = os.environ.get("NOTION_END_PROPERTY") or "end"
    hours_property = os.environ.get("NOTION_HOURS_PROPERTY") or "hours"

    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    url = f"https://api.notion.com/v1/databases/{database_id}/query"

    rows = []
    payload = {"page_size": 100}
    while True:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        rows.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        payload["start_cursor"] = data.get("next_cursor")

    records = []
    for row in rows:
        properties = row.get("properties", {})
        records.append(
            {
                "start": notion_date_value(properties, start_property),
                "end": notion_date_value(properties, end_property),
                "hours": notion_number_value(properties, hours_property),
            }
        )

    return clean_hours_dataframe(pd.DataFrame(records))


def load_data_from_csv() -> pd.DataFrame:
    data_path = PROJECT_DIR / "hours.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Export your Notion work-hours database there as hours.csv."
        )

    return clean_hours_dataframe(pd.read_csv(data_path))


def load_data() -> pd.DataFrame:
    if os.environ.get("NOTION_TOKEN") and os.environ.get("NOTION_DATABASE_ID"):
        return load_data_from_notion()
    return load_data_from_csv()



def load_vacations() -> list[dict]:
    vacations_path = PROJECT_DIR / "vacations.csv"
    if not vacations_path.exists():
        raise FileNotFoundError(
            "Missing vacations.csv. Create it in this folder with columns: name,start,end"
        )

    vacations_df = pd.read_csv(vacations_path)
    required_columns = {"name", "start", "end"}
    missing_columns = required_columns - set(vacations_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"vacations.csv is missing required column(s): {missing}")

    vacations = []
    for _, row in vacations_df.dropna(subset=["start", "end"]).iterrows():
        vacation = {
            "name": str(row["name"]).strip() if pd.notna(row["name"]) else "Vacation",
            "start": str(row["start"]).strip(),
            "end": str(row["end"]).strip(),
        }
        vacation["start_dt"] = pd.to_datetime(vacation["start"])
        vacation["end_dt"] = pd.to_datetime(vacation["end"])
        vacations.append(vacation)

    return vacations


def build_figure(df_all: pd.DataFrame, vacations: list[dict], year: Optional[int]) -> go.Figure:
    if year is None:
        first_date = df_all["start_dt"].min().normalize()
        last_date = df_all["start_dt"].max().normalize()
        year_start = pd.Timestamp(f"{first_date.year}-01-01")
        year_end = pd.Timestamp(f"{last_date.year}-12-31")
        title_period = f"{first_date.year}-{last_date.year}"
        show_rangeslider = True
        df = df_all.copy()
    else:
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        title_period = str(year)
        show_rangeslider = False
        df = df_all[df_all["start_dt"].dt.year == year].copy()

    display_vacations = [
        v for v in vacations if not (v["end_dt"] < year_start or v["start_dt"] > year_end)
    ]

    for v in vacations:
        mask = (df["start_dt"] >= v["start_dt"]) & (df["start_dt"] <= v["end_dt"])
        df = df[~mask]

    df["work_date"] = df["start_dt"].dt.normalize()
    df_daily = (
        df.groupby("work_date", as_index=False)
        .agg(
            first_start=("start_dt", "min"),
            last_end=("end_dt", "max"),
            daily_hours=("hours", "sum"),
        )
        .sort_values("work_date")
    )
    df_daily["arrival_hour"] = (
        (df_daily["first_start"] - df_daily["work_date"]).dt.total_seconds() / 3600
    )
    df_daily["leave_hour"] = (
        (df_daily["last_end"] - df_daily["work_date"]).dt.total_seconds() / 3600
    )
    df_daily["weekday"] = df_daily["work_date"].dt.weekday
    df_daily["week_start"] = (
        df_daily["work_date"] - pd.to_timedelta(df_daily["work_date"].dt.weekday, unit="D")
    )
    df_daily["is_weekend"] = df_daily["weekday"] >= 5
    df_daily["is_after_noon_arrival"] = (
        (~df_daily["is_weekend"]) & (df_daily["arrival_hour"] >= 12)
    )
    df_daily["special_day"] = df_daily["is_weekend"] | df_daily["is_after_noon_arrival"]
    df_daily["day_type"] = "Regular weekday"
    df_daily.loc[df_daily["is_weekend"], "day_type"] = "Weekend"
    df_daily.loc[df_daily["is_after_noon_arrival"], "day_type"] = "After-noon weekday arrival"
    df_daily.loc[
        df_daily["is_weekend"] & (df_daily["arrival_hour"] >= 12), "day_type"
    ] = "Weekend + after-noon arrival"

    df_daily_weekdays = df_daily[~df_daily["is_weekend"]].copy()
    df_daily_regular_weekdays = df_daily_weekdays[
        ~df_daily_weekdays["is_after_noon_arrival"]
    ].copy()
    arrival_colors = ["red" if special else "#1f77b4" for special in df_daily["special_day"]]
    leave_colors = ["red" if special else "#2ca02c" for special in df_daily["special_day"]]

    df["week_start"] = (
        df["start_dt"].dt.normalize() - pd.to_timedelta(df["start_dt"].dt.weekday, unit="D")
    )
    first_monday = year_start - pd.to_timedelta(year_start.weekday(), unit="D")
    last_monday = year_end - pd.to_timedelta(year_end.weekday(), unit="D")
    week_index = pd.date_range(start=first_monday, end=last_monday, freq="W-MON")
    df_weekly_full = pd.DataFrame({"week_start": week_index})

    df_weekly = df.groupby("week_start", as_index=False)["hours"].sum()
    df_weekly = (
        df_weekly_full.merge(df_weekly, on="week_start", how="left")
        .fillna({"hours": 0})
        .sort_values("week_start")
    )
    df_weekly["week_end"] = df_weekly["week_start"] + pd.Timedelta(days=6)
    df_weekly["week_mid"] = df_weekly["week_start"] + pd.Timedelta(days=3, hours=12)
    iso = df_weekly["week_start"].dt.isocalendar()
    df_weekly["iso_week"] = iso.week.astype(int)
    df_weekly["iso_year"] = iso.year.astype(int)
    df_weekly["has_vacation_overlap"] = False
    for v in vacations:
        overlap = (df_weekly["week_start"] <= v["end_dt"]) & (df_weekly["week_end"] >= v["start_dt"])
        df_weekly.loc[overlap, "has_vacation_overlap"] = True

    # Arrival/leave averages describe the normal weekday pattern, so weekends
    # and after-noon arrivals are shown but excluded from these timing averages.
    avg_arrival = df_daily_regular_weekdays["arrival_hour"].mean()
    avg_leave = df_daily_regular_weekdays["leave_hour"].mean()

    if not df_daily.empty:
        first_tracked_date = df["start_dt"].min().normalize()
        last_tracked_date = df["start_dt"].max().normalize()
        normalized_weeks = (
            df_daily.groupby("week_start", as_index=False)
            .agg(
                weekday_hours=(
                    "daily_hours",
                    lambda s: s[df_daily.loc[s.index, "weekday"] < 5].sum(),
                ),
                weekend_hours=(
                    "daily_hours",
                    lambda s: s[df_daily.loc[s.index, "weekday"] >= 5].sum(),
                ),
                worked_weekdays=(
                    "work_date",
                    lambda s: s[df_daily.loc[s.index, "weekday"] < 5].nunique(),
                ),
            )
            .sort_values("week_start")
        )
        normalized_weeks["week_end"] = normalized_weeks["week_start"] + pd.Timedelta(days=6)
        normalized_weeks["has_vacation_overlap"] = False
        for v in vacations:
            overlap = (
                (normalized_weeks["week_start"] <= v["end_dt"])
                & (normalized_weeks["week_end"] >= v["start_dt"])
            )
            normalized_weeks.loc[overlap, "has_vacation_overlap"] = True
        tracked_weeks = normalized_weeks[
            (normalized_weeks["week_start"] >= first_tracked_date)
            & (normalized_weeks["week_start"] <= last_tracked_date)
            & (~normalized_weeks["has_vacation_overlap"])
            & (normalized_weeks["worked_weekdays"] > 0)
        ].copy()
        tracked_weeks["adjusted_daily_hours"] = (
            tracked_weeks["weekday_hours"] / tracked_weeks["worked_weekdays"]
            + tracked_weeks["weekend_hours"] / 5
        )
        tracked_weeks["projected_hours"] = tracked_weeks["adjusted_daily_hours"] * 5
    else:
        tracked_weeks = pd.DataFrame(columns=["adjusted_daily_hours", "projected_hours"])
    avg_workday_hours = tracked_weeks["adjusted_daily_hours"].mean()
    avg_weekly = tracked_weeks["projected_hours"].mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.075,
        row_heights=[0.16, 0.42, 0.42],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
        subplot_titles=(
            f"Summary ({title_period})",
            f"Weekly Hours Overview ({title_period})",
            f"Arrival and Leaving Time at Work ({title_period})",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    stat_cards = [
        ("Typical Arrival", decimal_to_time_string(avg_arrival), "Regular weekdays"),
        ("Typical Leave", decimal_to_time_string(avg_leave), "Regular weekdays"),
        (
            "Avg Workday",
            f"{avg_workday_hours:.1f} hrs" if pd.notna(avg_workday_hours) else "N/A",
            "Weekend hours allocated",
        ),
        (
            "Weekly Avg",
            f"{avg_weekly:.1f} hrs" if pd.notna(avg_weekly) else "N/A",
            "5-day adjusted",
        ),
    ]
    card_width = 0.22
    card_gap = 0.025
    card_left = 0.02
    for idx, (label, value, note) in enumerate(stat_cards):
        x0 = card_left + idx * (card_width + card_gap)
        x1 = x0 + card_width
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=0.12,
            y1=0.88,
            xref="x domain",
            yref="y domain",
            fillcolor="#fbfdff",
            line=dict(color="#d7e0ec", width=1.2),
            layer="below",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=0.62,
            xref="x domain",
            yref="y domain",
            text=f"<b>{value}</b>",
            showarrow=False,
            font=dict(size=18, color="#1f2d4a"),
            yanchor="bottom",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=0.41,
            xref="x domain",
            yref="y domain",
            text=label,
            showarrow=False,
            font=dict(size=12, color="#53647f"),
            yanchor="middle",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=0.23,
            xref="x domain",
            yref="y domain",
            text=note,
            showarrow=False,
            font=dict(size=9, color="#7a8799"),
            yanchor="middle",
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[(x0 + x1) / 2],
                y=[0.50],
                mode="markers",
                marker=dict(size=110, color="rgba(0,0,0,0)", line=dict(width=0)),
                showlegend=False,
                hovertemplate=f"{label}<br>{value}<br>{note}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=df_weekly["week_mid"],
            y=df_weekly["hours"],
            customdata=df_weekly[["week_start", "week_end", "iso_week", "iso_year"]].to_numpy(),
            name="Weekly Total",
            marker_color="lightblue",
            opacity=0.9,
            width=7 * 24 * 60 * 60 * 1000,
            hovertemplate=(
                "Calendar Week: %{customdata[3]}-W%{customdata[2]}<br>"
                "Week: %{customdata[0]|%b %-d, %Y} to %{customdata[1]|%b %-d, %Y}<br>"
                "Hours: %{y:.1f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    if pd.notna(avg_weekly):
        fig.add_hline(
            y=avg_weekly,
            line_dash="dash",
            line_color="blue",
            row=2,
            col=1,
        )

    for v in display_vacations:
        fig.add_vrect(
            x0=v["start_dt"],
            x1=v["end_dt"],
            fillcolor="lightgrey",
            opacity=0.22,
            line_width=0,
            row=2,
            col=1,
        )

    label_base = df_weekly["hours"].max()
    if pd.isna(label_base) or label_base <= 0:
        label_base = 10

    label_x = []
    label_y = []
    label_text = []
    for i, v in enumerate(display_vacations):
        midpoint = v["start_dt"] + (v["end_dt"] - v["start_dt"]) / 2
        label_x.append(midpoint)
        label_y.append(label_base * (1.08 + 0.08 * (i % 2)))
        label_text.append(v["name"])

    fig.add_trace(
        go.Scatter(
            x=label_x,
            y=label_y,
            mode="text",
            text=label_text,
            textposition="top center",
            textfont=dict(size=11, color="dimgray"),
            name="Vacation",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    vacation_hover_top = max(label_y) if label_y else (df_weekly["hours"].max() * 1.2)
    if pd.isna(vacation_hover_top) or vacation_hover_top <= 0:
        vacation_hover_top = 10
    for v in display_vacations:
        mid = v["start_dt"] + (v["end_dt"] - v["start_dt"]) / 2
        duration_ms = max((v["end_dt"] - v["start_dt"]).total_seconds() * 1000, 1)
        fig.add_trace(
            go.Bar(
                x=[mid],
                y=[vacation_hover_top],
                width=[duration_ms],
                marker=dict(color="rgba(0,0,0,0)"),
                opacity=0,
                hovertemplate=(
                    f"Vacation: {v['name']}<br>"
                    f"Start: {v['start_dt'].strftime('%Y-%m-%d')}<br>"
                    f"End: {v['end_dt'].strftime('%Y-%m-%d')}<extra></extra>"
                ),
                showlegend=False,
                name="",
            ),
            row=2,
            col=1,
        )

    connector_x = []
    connector_y = []
    for _, row in df_daily.iterrows():
        connector_x.extend([row["work_date"], row["work_date"], None])
        connector_y.extend([row["arrival_hour"], row["leave_hour"], None])

    fig.add_trace(
        go.Scatter(
            x=connector_x,
            y=connector_y,
            mode="lines",
            name="Daily Span",
            line=dict(width=1.2, color="rgba(120,120,120,0.55)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_daily["work_date"],
            y=df_daily["arrival_hour"],
            customdata=df_daily[["first_start", "daily_hours", "day_type"]].to_numpy(),
            mode="markers",
            name="Arrival",
            marker=dict(size=5, color=arrival_colors),
            hovertemplate=(
                "Date: %{x|%b %-d, %Y}<br>"
                "Arrival: %{customdata[0]|%H:%M}<br>"
                "Worked: %{customdata[1]:.1f} hrs<br>"
                "Type: %{customdata[2]}<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )
    if pd.notna(avg_arrival):
        fig.add_hline(
            y=avg_arrival,
            line_dash="dash",
            line_width=2,
            line_color="#2c7fb8",
            row=3,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df_daily["work_date"],
            y=df_daily["leave_hour"],
            customdata=df_daily[["last_end", "daily_hours", "day_type"]].to_numpy(),
            mode="markers",
            name="Leave",
            marker=dict(size=5, color=leave_colors),
            hovertemplate=(
                "Date: %{x|%b %-d, %Y}<br>"
                "Leave: %{customdata[0]|%H:%M}<br>"
                "Worked: %{customdata[1]:.1f} hrs<br>"
                "Type: %{customdata[2]}<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )
    if pd.notna(avg_leave):
        fig.add_hline(
            y=avg_leave,
            line_dash="dash",
            line_width=2,
            line_color="#238443",
            row=3,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=7, color="red"),
            name="Weekend or after-noon arrival",
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )

    for v in display_vacations:
        fig.add_vrect(
            x0=v["start_dt"],
            x1=v["end_dt"],
            fillcolor="lightgrey",
            opacity=0.24,
            line_width=0,
            layer="below",
            row=3,
            col=1,
        )

    fig.update_layout(
        height=760,
        width=1040,
        autosize=True,
        title=f"Work Hours Overview for {title_period} (Vacation-Adjusted)",
        template="plotly_white",
        showlegend=True,
        bargap=0.02,
        dragmode="pan",
        margin=dict(t=78, r=92, l=52, b=34),
    )
    month_tick_format = "%b\n%Y" if (year_end - year_start).days > 370 else "%b"
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(row=2, col=1, tickformat=month_tick_format, dtick="M1", showticklabels=True)
    fig.update_xaxes(row=3, col=1, tickformat=month_tick_format, dtick="M1", matches="x")
    fig.update_xaxes(range=[year_start, year_end + pd.Timedelta(days=1)], row=2, col=1)
    fig.update_xaxes(rangeslider_visible=show_rangeslider, row=3, col=1)
    fig.update_yaxes(title_text="Hours", row=2, col=1)
    max_time = df_daily[["arrival_hour", "leave_hour"]].max().max() if not df_daily.empty else 24
    bottom_time = max(25, min(28, max_time + 1))
    fig.update_yaxes(title_text="Hour of Day", range=[bottom_time, 0], row=3, col=1)

    return fig


def main() -> None:
    df_all = load_data()
    vacations = load_vacations()

    docs_dir = PROJECT_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)

    for year in (2024, 2025, 2026):
        fig = build_figure(df_all=df_all, vacations=vacations, year=year)

        output_html = PROJECT_DIR / f"work_hours_{year}.html"
        docs_year_html = docs_dir / f"work_hours_{year}.html"
        fig.write_html(output_html.as_posix(), auto_open=False, config=PLOTLY_CONFIG)
        fig.write_html(docs_year_html.as_posix(), auto_open=False, config=PLOTLY_CONFIG)
        print(f"Saved: {output_html}")
        print(f"Saved: {docs_year_html}")

        if year == 2026:
            docs_index = docs_dir / "index.html"
            fig.write_html(docs_index.as_posix(), auto_open=False, config=PLOTLY_CONFIG)
            print(f"Saved: {docs_index}")

    fig_all = build_figure(df_all=df_all, vacations=vacations, year=None)
    output_all = PROJECT_DIR / "work_hours_all_years.html"
    docs_all = docs_dir / "work_hours_all_years.html"
    fig_all.write_html(output_all.as_posix(), auto_open=False, config=PLOTLY_CONFIG)
    fig_all.write_html(docs_all.as_posix(), auto_open=False, config=PLOTLY_CONFIG)
    print(f"Saved: {output_all}")
    print(f"Saved: {docs_all}")


if __name__ == "__main__":
    main()
