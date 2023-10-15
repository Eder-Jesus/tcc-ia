import pyodbc
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    url, usuario, senha, database = "107.23.75.36,1433", "sa", "Urubu100", "nutrilifelile"

    try:
        conn = pyodbc.connect(
            f'DRIVER=SQL Server;SERVER={url};DATABASE={database};UID={usuario};PWD={senha}'
        )
        cursor = conn.cursor()

        # Consulta para selecionar todos os usuários que desejam "Criar massa corporal" ou "Emagrecer"
        query_users = "SELECT * FROM Usuario WHERE objetivo IN ('Criar massa corporal', 'Emagrecer')"
        df_users = pd.read_sql(query_users, conn)

        # Consulta para selecionar todas as dietas (contém o ID do usuário e as respectivas receitas no campo listaReceita ("1,2")
        query_recipes = "SELECT * FROM Dieta"
        df_recipes = pd.read_sql(query_recipes, conn)

        conn.close()

        interactions_df = pd.merge(df_users, df_recipes, left_on='ID', right_on='ID_usuario')

        num_users = len(df_users)
        num_recipes = len(df_recipes)

        embedding_dim = 64

        X = interactions_df[['ID_usuario', 'ID_receita']].values
        y = interactions_df['classificacao'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        input_user = Input(shape=(1,))
        input_recipe = Input(shape=(1,))

        embedding_user = Embedding(input_dim=num_users, output_dim=embedding_dim)(input_user)
        embedding_recipe = Embedding(input_dim=num_recipes, output_dim=embedding_dim)(input_recipe)

        flatten_user = Flatten()(embedding_user)
        flatten_recipe = Flatten()(embedding_recipe)

        concatenated = Concatenate()([flatten_user, flatten_recipe])

        dense1 = Dense(128, activation='relu')(concatenated)
        dense2 = Dense(64, activation='relu')(dense1)
        output = Dense(1)(dense2)

        model = tf.keras.Model(inputs=[input_user, input_recipe], outputs=output)

        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=10, batch_size=64)

        loss = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
        print(f"Loss no conjunto de teste: {loss}")

        user_id = 1
        user_tensor = tf.convert_to_tensor([user_id])

        receitas_para_prever = df_recipes['ID'].values

        recipe_tensors = tf.convert_to_tensor(receitas_para_prever)

        predictions = model.predict([user_tensor, recipe_tensors])

        recomendacoes = list(zip(receitas_para_prever, predictions.flatten()))

        recomendacoes_ordenadas = sorted(recomendacoes, key=lambda x: x[1], reverse=True)

        top_10_recomendacoes = recomendacoes_ordenadas[:10]

        print("Top 10 recomendações:")
        for receita_id, pontuacao in top_10_recomendacoes:
            print(f"Receita ID: {receita_id}, Pontuação: {pontuação}")

    except pyodbc.Error as ex:
        print(f"Erro na conexão com o banco de dados: {ex}")
