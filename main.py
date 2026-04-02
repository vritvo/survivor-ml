from src.load import load_data
from src.features.build import build_modeling_table
from src.models.win import predict_season

def main():
    data = load_data()
    df = build_modeling_table(data)
    max_season = int(data["Episodes"][data["Episodes"]["version"] == "US"]["season"].max())
    for season in range(2, max_season + 1):
        print(f"\n=== Season {season}/{max_season} ===")
        predictions = predict_season(df, season)

if __name__ == "__main__":
    main()