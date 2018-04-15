### 常用敘述性統計

* 連續資料與類別資料
* 資料性質:
    * 平均 (算術平均, 加權平均, 幾何平均, 調和平均)
    * 中位數
    * 眾數
    * 全距與四分位差
    * 標準差 (樣本標準差與母體標準差)
    * 變異數 (Variance, 方差)
    * 變異係數 (C.V.)
    * 共變異數 (Co-Variance, 斜方差)
* 連續分布
    * 常態分布
* 離散機率分布
    * 二項式分布
    * 卜瓦松分布

### 常用假說性檢定

![](data/images/statistics.png =100%x*)

#### 註解

1. ANOVA 並無法得知各組間的比較，如 (Sample1, Sample2)、(Sample1, Sample3)、...。透過 Multiple comparison 可以進行 pairwise 的比較而挑出各組間是否有顯著差異，常用的方法如 Tukey's method、Fisher's Least Significant Difference 與 Duncan's New Multiple Range Test 等。
2. 常見的 correltion 方式如 Yate's correlation 等。
3. Chi-square test 與 Fisher's exact test 主要差異為 2x2 相關聯表中的數值大或小 (或者說 frequency 大小)。一般而言，Fisher's exact test 更常用於數值小的 p-value 計算，且較為合理。The test based on the hypergeometric distribution (hypergeometric test) is identical to the corresponding one-tailed version of Fisher's exact test. Reciprocally, the p-value of a two-sided Fisher's exact test can be calculated as the sum of two appropriate hypergeometric tests.
4. 當比較兩個未成對的群組時 (two unpaired groups)，在有參數(parametric)/常態假說/中央極限定理下，應使用未成對 t 檢定 (unpaired t-test)。

#### 各項檢定

* 假說性檢定
* 信賴區間
* 檢定分布
    * 檢定常態分布: Shapiro-Wilk Test
* 標準化 (Standardisation): 目的使樣本內的資料間可以比較
* 正規化 (Normalisation): 目的使含有資料的樣本間可以比較
    * 無母數
        * Quantile Normalization, [Rscript](data/Quantile_Normalization_R.html)
* 相關分析
    * 連續資料
        * 有母數: Pearson's Correlation, [Rscript](data/Correlation_R.html)
        * 無母數: Spearman's Correlation, [Rscript](data/Correlation_R.html)
    * 類別資料
        * 兩個類別: Logistic Regression, [Rscript](data/Logistic_Regression_R.html)
        * 三類以上: Multinomial/Ordinal Logistic Regression,  [Rscript](data/Multinomial_Log-linear_Models_R.html)
* 兩組資料
    * 連續資料
        * 有母數
            * 獨立樣本: Independent t-test
            * 相依樣本: Paired t-test
        * 無母數
            * 獨立樣本: Wilcoxon-Mann-Whitney Two-Sample Test
            * 相依樣本: Wilcoxon Singed-Rank Test
    * 類別資料
        * 兩個類別
            * Chi-square test
            * Fisher's exact test: [Rscript](data/Fisher_Exact_Test_R.html)
            * Hypergeometeric test: [Rscript](data/Hypergeometeric_test_R.html)
* 三組或以上資料
    * 連續資料
        * 有母數
            * 獨立樣本: 
                * ANOVA (F-test)
                * Fisher's Least Significant Difference (LSD)
                * Duncan's New Multiple Range Test





