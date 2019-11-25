class CleanData:
    def ___init__(self, name):
        self.name = name

    def strip_columns(self, df):
        df = df.rename(columns=lambda x: x.strip())
        return df
    
        