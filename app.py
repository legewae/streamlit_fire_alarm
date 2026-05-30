import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Настройка широкого формата страницы (опционально)
st.set_page_config(page_title="Дашборд ML", layout="wide")

# --- БОКОВАЯ ПАНЕЛЬ НАВИГАЦИИ ---
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ("1. О разработчике", "2. Информация о данных", "3. Визуализация данных", "4. Инференс моделей")
)

# --- СТРАНИЦА 1: О разработчике ---
if page == "1. О разработчике":
    st.title("Информация о разработчике")
    st.write("**ФИО:** Гришанов Егор Викторович")
    st.write("**Учебная группа:** ФИТ-242")
    st.write("**Тема РГР:** Разработка Web-приложения (дашборда)для инференса (вывода) моделей ML и анализа данных")
    


elif page == "2. Информация о данных":
    st.title("Описание набора данных")
    st.write("Для работы был выбран датасет классификации, содержащий данные о разных метриках состояния воздуха для предсказания пожара(бинарная классификация)")
    st.write("В ходе предобработки и EDA-анализа были удалены признаки, не несущие полезной информации для модели")
    st.write("Выяснилось, что все записи в датасете разделены на сессии работы датчика. Используя данную информацию, были заполнены пропуски линейной интерполяцией. Сделано это было для того, чтобы пропуск не мог быть заполнен информацией датчика с совершенно другого временного промежутка.")
    st.write("Датасет имеет следующие признаки(12 + 1 целевой):")
    text = """
- Temperature[C]: Температура в цельсиях
- Humidity[%]: Процент влажности воздуха
- TVOC[ppb]: Общая концентрация летучих органических соединений в частях на миллиард
- eCO2[ppm]: Общая концентрация углекислого газа на миллион
- Raw H2: Сырое значение концентрации водорода
- Raw Ethanol: Сырое значение концентрации этанола
- Pressure[hPa]: Атмосферное давление в гектопаскалях
- PM1.0: Концентрация частиц размером менее 1.0 микрона
- PM2.5: Концентрация частиц размером менее 2.5 микрона в воздухе
- NC0.5: Концентрация частиц размером 0.5 мкм и более
- NC1.0: Концентрация частиц размером 1 мкм и более
- NC2.5: Концентрация частиц размером 2.5 мкм и более
- Fire Alarm: Сработал ли датчик (целевой признак)"""
    st.write(text)

    st.write("Весь обработанный датасет:")
    df = pd.read_csv('data/classification_processed.csv')
    st.dataframe(df)

# --- СТРАНИЦА 3: Визуализация ---
elif page == "3. Визуализация данных":
    st.title("Визуализация зависимостей")
    st.subheader("Корреляционная матрица признаков расположена ниже:")

    df = pd.read_csv('data/classification_processed.csv')

    plt.figure(figsize=(14, 12))
    correlation_matrix = df.corr()

    sns.heatmap(correlation_matrix, 
            cmap="coolwarm", 
            center=0, 
            annot=True, 
            fmt='.2f',
            square=True,
            linewidths=0.5)
 
    plt.title("Корреляционная матрица признаков", fontsize=16, pad=20)
    plt.tight_layout()
    st.pyplot(plt,width = 1000)

    st.subheader("Распределение целевого признака можно увидеть на круговой диаграмме:")
    plt.figure(figsize=(8, 8))
    counts = df['Fire Alarm'].value_counts()
    plt.pie(counts.values, labels=['Пожар(1)', 'Нет пожара(0)'], autopct='%1.1f%%')
    plt.title('Распределение целевого признака Fire Alarm')
    st.pyplot(plt, width = 800)

    st.subheader("Распределение признаков без выбросов")
    num_df = df.drop(columns = ['Fire Alarm'])

    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    IQR = q3-q1

    data_filtered = num_df[~((num_df < (q1 - 1.5 * IQR)) |(num_df > (q3 + 1.5 * IQR))).any(axis=1)]

    data_filtered.hist(figsize=(16,9), bins=50)
    plt.suptitle("Распределения признаков без выбросов")
    st.pyplot(plt, width = 1000)

    st.subheader("Сравнение средних значений признаков: Пожар (1) vs Норма (0)")

    avg_data = df.groupby('Fire Alarm').mean().T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax1.set_xscale('log')
    ax2.set_xscale('log')
    # Левый график (Fire Alarm = 0)
    bar1 = ax1.barh(avg_data.index, avg_data[0], color='skyblue')
    ax1.set_title('Средние значения (Fire Alarm = 0)')
    ax1.set_xlabel('Среднее значение')
    ax1.bar_label(bar1, padding=5, fmt='%.1f', fontsize=8)

    # Правый график (Fire Alarm = 1)
    bar2 = ax2.barh(avg_data.index, avg_data[1], color='salmon')
    ax2.set_title('Средние значения (Fire Alarm = 1)')
    ax2.set_xlabel('Среднее значение')
    ax2.bar_label(bar2, padding=5, fmt='%.1f', fontsize=8)
    
    plt.tight_layout()

    st.pyplot(fig)

elif page == "4. Инференс моделей":
    st.title("Предсказание модели (Инференс)")

    scaler = None
    with open("models/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    @st.cache_resource
    def get_model(name):
        model_paths = {
            "AdaClassifier(DecisionTreeClassifier)": "models/AdaClassifier.pkl",
            "Случайный лес": "models/RandomForestClassifier.pkl",
            "LGMClassifier": "models/LGMClassifier.txt",
            "KNN": "models/knn_model.pkl",
            "StackingClassifier(RandomF., SVM, KNN)": "models/StackingClassifier.pkl"
        }

        if name == "LGMClassifier":
                return lgb.Booster(model_file=model_paths[name])
        
        with open(model_paths[name], 'rb') as f:
            
            return pickle.load(f)


    st.subheader("Введите параметры датчиков:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.number_input("Temperature [C]", value=20.0)
        humidity = st.number_input("Humidity [%]", value=50.0)
        tvoc = st.number_input("TVOC [ppb]", value=0.0)
        eco2 = st.number_input("eCO2 [ppm]", value=400.0)
    
    with col2:
        raw_h2 = st.number_input("Raw H2", value=10000.0)
        raw_ethanol = st.number_input("Raw Ethanol", value=20000.0)
        pressure = st.number_input("Pressure [hPa]", value=930.0)
        PM1 = st.number_input("PM1.0")
        
    with col3:
        PM2_5 = st.number_input("PM2.5")
        NC0_5 = st.number_input("NC0.5")
        NC1_0 = st.number_input("NC1.0")
        NC2_5 = st.number_input("NC2.5")


    model_choice = st.selectbox("Выберите модель:", ["AdaClassifier(DecisionTreeClassifier)", "Случайный лес", "LGMClassifier", "KNN", "StackingClassifier(RandomF., SVM, KNN)"])
    
    if st.button("Сделать предсказание!"):

        input_data = pd.DataFrame({
            'Temperature[C]': [temp],
            'Humidity[%]': [humidity],
            'TVOC[ppb]': [tvoc],
            'eCO2[ppm]': [eco2],
            'Raw H2': [raw_h2],
            'Raw Ethanol': [raw_ethanol],
            'Pressure[hPa]': [pressure],
            'PM1.0': [PM1],
            'PM2.5': [PM2_5],
            'NC0.5': [NC0_5],
            'NC1.0': [NC1_0],
            'NC2.5': [NC2_5]
        })
        
        # Загружаем модель
        try:
            model = get_model(model_choice)

            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)
            if model_choice == "LGMClassifier":
                prediction = [1 if pred > 0.5 else 0 for pred in prediction]
                
            # Вывод результата
            if prediction[0] == 1:
                st.error("ПОЖАР (1)")
            else:
                st.success("ПОЖАРА НЕТ (0)")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

    st.subheader("Загрузите файл для пакетного предсказания")
    uploaded_file = st.file_uploader("Загрузить *.csv", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Превью загруженных данных:")
        st.dataframe(batch_df.head())
        
        model_choice_batch = st.selectbox("Выберите модель:", ["AdaClassifier(DecisionTreeClassifier)", "Случайный лес", "LGMClassifier", "KNN", "StackingClassifier(RandomF., SVM, KNN)"], key = 3)
        
        if st.button("Предсказать для всего файла!"):
            try:
                model_batch = get_model(model_choice_batch)
                
                # Применяем scaler к загруженным данным
                scaled_batch = scaler.transform(batch_df)
                batch_predictions = model_batch.predict(scaled_batch)

                if model_choice_batch == "LGMClassifier":
                    batch_predictions = [1 if pred > 0.5 else 0 for pred in batch_predictions]
                
                # Создаем копию датафрейма и добавляем колонку с предсказаниями
                result_df = batch_df.copy()
                result_df['Fire Alarm_Prediction'] = batch_predictions
                
                st.success("Предсказание успешно выполнено!")
                st.dataframe(result_df)
                
                # возможность скачать файл
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать результаты",
                    data=csv_data,
                    file_name='predictions_result.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}. Убедитесь, что структура колонок совпадает с оригинальным датасетом.")
