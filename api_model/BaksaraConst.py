from tensorflow import keras

MODEL_PATH = './save_model/model.h5'
# MODEL_PATH = './api_model/save_model/model.h5'
MODEL = keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['carakan_ba', 'carakan_ca', 'carakan_da', 'carakan_dha', 'carakan_ga', 'carakan_ha',
                    'carakan_ja', 'carakan_ka', 'carakan_la', 'carakan_ma', 'carakan_na', 'carakan_nga',
                    'carakan_nya', 'carakan_pa', 'carakan_ra', 'carakan_sa', 'carakan_ta', 'carakan_tha',
                    'carakan_wa', 'carakan_ya', 'sandhangan_e', 'sandhangan_e2', 'sandhangan_h', 'sandhangan_i',
                    'sandhangan_ng', 'sandhangan_o', 'sandhangan_r', 'sandhangan_u']