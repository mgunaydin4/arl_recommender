#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")


#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.

df_ = pd.read_csv("Case Study 1/ArmutARL-221114-234936/armut_data.csv")
df = df_.copy()
df.head()
# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)


# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()
df.info()

# strftime() => tarih ve saati string ifadelere dönüştürür!!!!!

df["new_date"] = df["CreateDate"].dt.strftime("%Y-%m")

df["SepetId"] = df["UserId"].astype(str) + "_" + df["new_date"]
df.head()

#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

invoice_product_df = df.groupby(["SepetId", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()

# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_items = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_items, metric="support", min_threshold=0.01)
rules.head()

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommendation(df, hizmet, min_support=0.01, metric="support", min_threshold=0.01, rec_count=1):
    from mlxtend.frequent_patterns import apriori, association_rules

    # Sepet-hizmet pivot table oluşturma
    invoice_product_df = df.groupby(["SepetId", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(
        lambda x: 1 if x > 0 else 0)

    # Apriori algoritması ile sık hizmet kümelerini bulma
    frequent_items = apriori(invoice_product_df, min_support=min_support, use_colnames=True)

    # Birliktelik kurallarını oluşturma
    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)

    # Seçilen hizmeti içeren kuralları filtrele
    sorted_rules = rules[rules["antecedents"].apply(lambda x: hizmet in x)].sort_values("lift", ascending=False)

    # Önerilecek hizmet sayısı kadar öneriyi al
    recommendations = sorted_rules["consequents"].apply(lambda x: list(x)[0]).head(rec_count)

    return recommendations


# Fonksiyonu çağırarak 2_0 hizmetine göre önerilerde bulunma
recommended_services = arl_recommendation(df, "2_0", rec_count=5)
print(recommended_services)
