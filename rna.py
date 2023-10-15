import pyodbc
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Multiply
from sklearn.model_selection import train_test_split
import numpy as np

def recommend_recipes_for_user(user_id, user_objetivo, model, df_recipes, top_n=10):

    user_tensor = tf.convert_to_tensor([user_id])

    num_samples = len(df_recipes)
    user_tensor = tf.tile(user_tensor, [num_samples])

    recipe_ids = df_recipes['ListaReceita'].tolist()
    recipe_tensors = tf.convert_to_tensor(recipe_ids, dtype=tf.int32)

    predictions = model.predict([user_tensor, recipe_tensors])

    recomendacoes = list(zip(recipe_ids, predictions.flatten()))

    recomendacoes_ordenadas = sorted(recomendacoes, key=lambda x: x[1], reverse=True)

    top_recomendacoes = recomendacoes_ordenadas[:top_n]

    return top_recomendacoes

if __name__ == "__main__":
    url, usuario, senha, database = "107.23.75.36,1433", "sa", "Urubu100", "nutrilifelile"

    try:
        conn = pyodbc.connect(
            f'DRIVER=SQL Server;SERVER={url};DATABASE={database};UID={usuario};PWD={senha}'
        )
        cursor = conn.cursor()

        user_ids = [386, 387, 388, 390, 391, 393, 395, 396, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413]
        query_users = "SELECT * FROM Usuario WHERE ID IN ({0})".format(
            ", ".join(map(str, user_ids)))
        df_users = pd.read_sql(query_users, conn)

        query_recipes = "SELECT * FROM Dieta"
        df_recipes = pd.read_sql(query_recipes, conn)

        conn.close()

        df_recipes['ListaReceita'] = df_recipes['ListaReceita'].apply(lambda x: list(map(int, x.split(','))))

        df_recipes = df_recipes.explode('ListaReceita')

        interactions_df = pd.merge(df_users, df_recipes, left_on='Id', right_on='UsuarioId')

        num_users = len(df_users)
        num_recipes = len(df_recipes)

        embedding_dim = 64

        X = interactions_df[['UsuarioId', 'ListaReceita']].values
        y = interactions_df['Objetivo'].values

        y = pd.Series(y).map({"Emagrecer": 0, "Criar massa corporal": 1}).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        max_user_id = max(X_train[:, 0])
        max_recipe_id = max(X_train[:, 1])

        X_test = X_test[(X_test[:, 0] <= max_user_id) & (X_test[:, 1] <= max_recipe_id)]
        y_test = y_test[(X_test[:, 0] <= max_user_id) & (X_test[:, 1] <= max_recipe_id)]

        input_user = Input(shape=(1,))
        input_recipe = Input(shape=(1,))

        embedding_user = Embedding(input_dim=max_user_id + 1, output_dim=embedding_dim)(input_user)
        embedding_recipe = Embedding(input_dim=max_recipe_id + 1, output_dim=embedding_dim)(input_recipe)

        multiply_layer = Multiply()([embedding_user, embedding_recipe])

        flatten = Flatten()(multiply_layer)

        dense1 = Dense(128, activation='relu')(flatten)
        dense2 = Dense(64, activation='relu')(dense1)
        output = Dense(1, activation='sigmoid')(dense2)

        model = tf.keras.Model(inputs=[input_user, input_recipe], outputs=output)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        user_ids_tensor = tf.convert_to_tensor(X_train[:, 0], dtype=tf.int32)
        recipe_ids_tensor = tf.convert_to_tensor(X_train[:, 1], dtype=tf.int32)

        model.fit([user_ids_tensor, recipe_ids_tensor], y_train, epochs=10, batch_size=64)

        user_ids_test = tf.convert_to_tensor(X_test[:, 0], dtype=tf.int32)
        recipe_ids_test = tf.convert_to_tensor(X_test[:, 1], dtype=tf.int32)

        loss, accuracy = model.evaluate([user_ids_test, recipe_ids_test], y_test)
        print(f"Loss no conjunto de teste: {loss}")
        print(f"Acurácia no conjunto de teste: {accuracy}")

        user_id = 386

        user_objetivo = df_users[df_users['Id'] == user_id]['Objetivo'].values[0]

        top_10_recomendacoes = recommend_recipes_for_user(user_id, user_objetivo, model, df_recipes, top_n=10)

        print(f"Principais recomendações para o usuário {user_id} com objetivo '{user_objetivo}':")
        for receita_id, pontuacao in top_10_recomendacoes:
            print(f"Receita ID: {receita_id}, Pontuação: {pontuacao}")

    except pyodbc.Error as ex:
        print(f"Erro na conexão com o banco de dados: {ex}")