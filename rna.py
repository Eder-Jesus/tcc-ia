import pyodbc
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Multiply
from sklearn.model_selection import train_test_split
import numpy as np
import time

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
    url, usuario, senha, database = "18.233.72.114,1433", "sa", "Urubu100", "nutrilifelile"

    while True:
        try:
            conn = pyodbc.connect(
                f'DRIVER=SQL Server;SERVER={url};DATABASE={database};UID={usuario};PWD={senha}'
            )
            cursor = conn.cursor()

            user_ids = [363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381]
            query_users = "SELECT * FROM Usuario WHERE ID IN ({0})".format(
                ", ".join(map(str, user_ids)))
            df_users = pd.read_sql(query_users, conn)

            query_recipes = "SELECT * FROM Dieta"
            df_recipes = pd.read_sql(query_recipes, conn)

            query_recipes = "SELECT TOP 1 * FROM Usuario where status = 1"
            df_user = pd.read_sql(query_recipes, conn)

            print(df_user)
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

            model.fit([user_ids_tensor, recipe_ids_tensor], y_train, epochs=100, batch_size=64)

            user_id = 362
            userTeste = df_user['Id'].values[0]
            userObjetivo = df_user['Objetivo'].values[0]
            print(userObjetivo)

            top_10_recomendacoes = recommend_recipes_for_user(user_id, userObjetivo, model, df_recipes, top_n=50)

            ids_unicos = set(receita_id for receita_id, _ in top_10_recomendacoes)

            ids_unicos = list(ids_unicos)[:10]

            ids_string = ""

            for receita_id in ids_unicos:
                ids_string += str(receita_id) + ","

            if ids_string.endswith(","):
                ids_string = ids_string[:-1]

            queryInsert = f"INSERT into Dieta (UsuarioId, ListaReceita) values ({userTeste}, '{ids_string}')"
            queryUpdate = f"UPDATE Usuario SET Status = 0 WHERE id = {userTeste}"

            cursor.execute(queryInsert)
            conn.commit()
            cursor.execute(queryUpdate)
            conn.commit()

            print(ids_unicos)
            print(f"Principais recomendações para o usuário {userTeste} com objetivo '{userObjetivo}':")
            for receita_id, pontuacao in top_10_recomendacoes:
                print(f"Receita ID: {receita_id}, Pontuação: {pontuacao}")

            conn.close()

        except pyodbc.Error as ex:
            print(f"Erro na conexão com o banco de dados: {ex}")

        time.sleep(10)
