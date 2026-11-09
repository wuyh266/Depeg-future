import pandas as pd
import numpy as np
import os
try:
    from tools import save_csv_with_backup
except ImportError:
    from src.tools import save_csv_with_backup


class DepegLabeler:
    def __init__(self, threshold_pct=5, min_duration_hours=6):
        self.threshold_pct = threshold_pct
        self.min_duration_hours = min_duration_hours

    def compute_deviation(self, df):
        df = df.copy()
        if 'anchor' not in df.columns:
            df['anchor'] = 1.0
        df['anchor_deviation'] = abs(df['price'] - df['anchor']) * 100
        df['is_depegged'] = df['anchor_deviation'] > self.threshold_pct
        return df

    def label_timeseries(self, df):
        df = self.compute_deviation(df)
        df['depeg_label'] = df['is_depegged'].astype(int)
        return df

    def detect_events(self, df):
        df = df.copy()
        df = self.compute_deviation(df)
        
        events = []
        in_event = False
        event_start = None
        
        for i, row in df.iterrows():
            if row['is_depegged'] and not in_event:
                in_event = True
                event_start = i
                event_max_dev = row['anchor_deviation']
            elif row['is_depegged'] and in_event:
                if row['anchor_deviation'] > event_max_dev:
                    event_max_dev = row['anchor_deviation']
            elif not row['is_depegged'] and in_event:
                duration_hours = (df.loc[i, 'timestamp'] - df.loc[event_start, 'timestamp']).total_seconds() / 3600
                if duration_hours >= self.min_duration_hours:
                    events.append({
                        'start_idx': event_start,
                        'end_idx': i,
                        'start_time': df.loc[event_start, 'timestamp'],
                        'end_time': df.loc[i, 'timestamp'],
                        'duration_hours': duration_hours,
                        'max_deviation': event_max_dev
                    })
                in_event = False
                event_start = None
        
        if in_event:
            duration_hours = (df.iloc[-1]['timestamp'] - df.iloc[event_start]['timestamp']).total_seconds() / 3600
            if duration_hours >= self.min_duration_hours:
                events.append({
                    'start_idx': event_start,
                    'end_idx': len(df) - 1,
                    'start_time': df.loc[event_start, 'timestamp'],
                    'end_time': df.iloc[-1]['timestamp'],
                    'duration_hours': duration_hours,
                    'max_deviation': df.loc[event_start:, 'anchor_deviation'].max()
                })
        
        return events

    def generate_future_labels(self, df, horizon_hours=24):
        df = df.copy()
        df = self.compute_deviation(df)
        
        labels_6h = []
        labels_24h = []
        labels_168h = []
        
        for i in range(len(df)):
            future_6h = df.iloc[i+1:min(i+7, len(df))]
            future_24h = df.iloc[i+1:min(i+25, len(df))]
            future_168h = df.iloc[i+1:min(i+169, len(df))]
            
            label_6h = 1 if len(future_6h) > 0 and future_6h['anchor_deviation'].max() > self.threshold_pct else 0
            label_24h = 1 if len(future_24h) > 0 and future_24h['anchor_deviation'].max() > self.threshold_pct else 0
            label_168h = 1 if len(future_168h) > 0 and future_168h['anchor_deviation'].max() > self.threshold_pct else 0
            
            time_to_depeg_6h = None
            time_to_depeg_24h = None
            time_to_depeg_168h = None
            
            if label_6h:
                depeg_idx = future_6h[future_6h['is_depegged']].index[0]
                time_to_depeg_6h = (df.loc[depeg_idx, 'timestamp'] - df.loc[i, 'timestamp']).total_seconds() / 3600
            
            if label_24h:
                depeg_idx = future_24h[future_24h['is_depegged']].index[0]
                time_to_depeg_24h = (df.loc[depeg_idx, 'timestamp'] - df.loc[i, 'timestamp']).total_seconds() / 3600
            
            if label_168h:
                depeg_idx = future_168h[future_168h['is_depegged']].index[0]
                time_to_depeg_168h = (df.loc[depeg_idx, 'timestamp'] - df.loc[i, 'timestamp']).total_seconds() / 3600
            
            labels_6h.append(label_6h)
            labels_24h.append(label_24h)
            labels_168h.append(label_168h)
        
        df['label_6h'] = labels_6h
        df['label_24h'] = labels_24h
        df['label_168h'] = labels_168h
        
        df['time_to_depeg_6h'] = [None] * len(df)
        df['time_to_depeg_24h'] = [None] * len(df)
        df['time_to_depeg_168h'] = [None] * len(df)
        
        return df

    def process_coin(self, coin_name, data_dir='data'):
        price_file = os.path.join(data_dir, coin_name.lower(), 'prices.csv')
        
        if not os.path.exists(price_file):
            print(f"Price file not found for {coin_name}")
            return None, []
        
        df = pd.read_csv(price_file, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if 'anchor' not in df.columns:
            df['anchor'] = 1.0
        
        df = self.generate_future_labels(df, horizon_hours=24)
        events = self.detect_events(df)
        
        label_file = os.path.join(data_dir, coin_name.lower(), 'depeg_labels.csv')
        backup_path, label_file = save_csv_with_backup(df, label_file, index=False)
        if backup_path:
            print(f"{coin_name}: Existing labels backed up to {backup_path}")
        
        event_file = os.path.join(data_dir, coin_name.lower(), 'depeg_events.csv')
        events_df = pd.DataFrame(events)
        if len(events_df) > 0:
            backup_path, event_file = save_csv_with_backup(events_df, event_file, index=False)
            if backup_path:
                print(f"{coin_name}: Existing event file backed up to {backup_path}")
        
        print(f"{coin_name}: Found {len(events)} depeg events, saved labels")
        return df, events


def main():
    import fetch_price
    
    fetcher = fetch_price.PriceFetcher()
    
    STABLECOINS = ['USDC', 'USDT', 'DAI', 'UST', 'FRAX', 'TUSD']
    
    for coin in STABLECOINS:
        print(f"\nProcessing {coin}...")
        
        df = fetcher.fetch_historical(coin, days=365*3)
        
        if df is not None:
            labeler = DepegLabeler(threshold_pct=5, min_duration_hours=6)
            df, events = labeler.process_coin(coin, data_dir='data')
            
            if events:
                print(f"  Events detected:")
                for e in events:
                    print(f"    {e['start_time']} - {e['end_time']}: max deviation {e['max_deviation']:.2f}%")


if __name__ == '__main__':
    main()
