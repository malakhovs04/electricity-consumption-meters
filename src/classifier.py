from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class DataSplitter:

    def split(self, df):

        X = df.drop(columns=['meter_id', 'consumer_class'])
        y = df["consumer_class"]
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


class ConsumerClassifier:

    def train(self, X_train, X_test, y_train, y_test):

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\n===== CLASSIFICATION REPORT =====")
        print(classification_report(y_test, y_pred))

        return model, y_pred