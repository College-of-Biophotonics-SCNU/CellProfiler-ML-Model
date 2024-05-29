# CellProfiler基本使用
CellProfiler可以用于药物筛选，利用预先定义的手工特征进行特征提取，将特征利用基本的机器学习进行拟合，精准的判断属于哪种化合物影响的细胞反应。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716204602767-a8e78cd7-6ff6-4e6a-a6c7-4531413d9577.png#averageHue=%23625d21&clientId=u4e9dc7c5-4cce-4&from=paste&height=843&id=u39ef3311&originHeight=1264&originWidth=2230&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=1982927&status=done&style=none&taskId=udf34a757-e57d-4647-b552-70636490853&title=&width=1486.6666666666667)
# 文件metadata的数据导出
为了方便文件的读取，尤其是CellProfiler以及后续的深度网络计算，这里采用了文件命名的方式存储图像中的细节信息。
如 **_clMcf7_exp20240515_h2_b2_ic1_cAA.tif _**，其对应含义如下：

- cl表示Cell Line细胞系类型，
- exp表示实验批次，利用时间进行表示，
- h表示时间点，
- b表示batch表示批次，相当于拍照的孔板批次，
- ic表示is control表示是否为控制组还是对照组，该参数类似于label标签定义，
- c表示channel通道定义，由于该模型具有FRET和MHCS所以 AA、DD、DA表示FRET通道定义，DNA，Nucleus等表示多亚细胞器的通道定义，
- .tiff 表示对应的图像信息

转换实验室图像存储名称格式的程序代码存放在在**_dataloader_**文件包内。
目前图像数据分为两类，一个是FRET图像以及明场BF图像
## FRET图像
目前具有6个通道数据，AA、DD、DA以及MB、ED、Rc

- MB 是AA、DD、DA三个通道图像交集得出的蒙版数据
- ED是计算FRET效率得出来的数据
- Rc是对应FRET蛋白的浓度比图像分布

![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716951024126-2f609db0-f04f-4b80-8490-dbbe0bcb57c2.png#averageHue=%23f4f1ed&clientId=u2e573a1d-007c-4&from=paste&height=1012&id=u359e1429&originHeight=1518&originWidth=3292&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=287780&status=done&style=none&taskId=ud9cf60dc-14ce-4dde-a635-1adab8036e3&title=&width=2194.6666666666665)
## 明场图像
BF明场图像
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716990548937-3a35e104-8126-4a4e-b745-499e76669ea2.png#averageHue=%234a4a4a&clientId=uc035a44c-201b-4&from=paste&height=334&id=u90569723&originHeight=501&originWidth=501&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=234889&status=done&style=none&taskId=ue9a97993-29f4-406f-a4e1-9912edb8715&title=&width=334)
# 预处理操作
## 高斯平滑
高斯滤波器，高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。将噪声减弱。
CellProfiler采用GaussianFilter模块进行处理。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716991013548-35b78028-036c-48e6-8184-c1e0106d2790.png#averageHue=%234e4e4e&clientId=uc035a44c-201b-4&from=paste&height=375&id=uf4f59479&originHeight=562&originWidth=1132&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=123072&status=done&style=none&taskId=u0c5f5cf8-2f88-470b-b3f3-c7f11e8d7a8&title=&width=754.6666666666666)
## 图像去噪
降低噪声。该模块执行非局部的降噪方法。这将运行一个5x5像素的补丁，最大距离为2像素，以搜索用于使用0.2的截止点去噪的补丁。ReduceNoise执行非局部均值降噪。不像在GaussianFilter中那样，只使用中心像素周围的像素邻域进行去噪，而是将多个邻域合并在一起。通过使用相关度量和截止值扫描图像以寻找与中心像素周围区域相似的区域来确定邻域池。
CellProfiler采用ReduceNoise模块。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716991196115-4201af6d-533c-467f-85a1-5daad850842f.png#averageHue=%23525252&clientId=uc035a44c-201b-4&from=paste&height=411&id=ua4100957&originHeight=616&originWidth=1194&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=116955&status=done&style=none&taskId=u84e8c3c2-a8bc-4c8f-ac4a-1aa6009b09d&title=&width=796)
## 单细胞分割
利用DD图像进行细胞区域的获取，下图为2h的细胞图像
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716993479680-b39508ad-4367-40fa-8e7f-beda968d0d6f.png#averageHue=%23a8a7a7&clientId=uc035a44c-201b-4&from=paste&height=440&id=uc75c2f1e&originHeight=660&originWidth=934&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=150896&status=done&style=none&taskId=ua1f7b879-511b-4d24-a627-beebf6de1ba&title=&width=622.6666666666666)
对于6h图像分割结果
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716993700800-5660ef8d-7d57-4d6a-ae66-bb9d1f66778c.png#averageHue=%23a7a6a5&clientId=uc035a44c-201b-4&from=paste&height=436&id=uc0bc0a5e&originHeight=654&originWidth=913&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=172427&status=done&style=none&taskId=u2c0411b5-96ac-4be7-9b29-b9f9a25880a&title=&width=608.6666666666666)
## 明场图像增强
将明场按照纹理进行提取，拿到纹理信息方便后面进行信息提取
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716274592973-54a9350c-9fec-4797-9ee3-a253ae44fbe0.png#averageHue=%23acacac&clientId=u4e9dc7c5-4cce-4&from=paste&height=363&id=u62d01b88&originHeight=544&originWidth=945&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=166067&status=done&style=none&taskId=u116c63fd-346f-4cae-9d08-32b23ba7fb7&title=&width=630)
## 掩码无用信息
使用MB图像，屏蔽除了细胞以外无用的背景信息信息。
CellProfiler采用了MaskImage模块。利用Muban的0-1区域图像屏蔽掉不需要的明场图像。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716274670398-5a31d169-fa44-4f2a-9258-e632e01b62a0.png#averageHue=%236f6f6f&clientId=u4e9dc7c5-4cce-4&from=paste&height=292&id=u44ecaaf6&originHeight=438&originWidth=946&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=134435&status=done&style=none&taskId=ucd4ad7e5-3cd2-4392-9daf-9c026000057&title=&width=630.6666666666666)
## 分割明场区域内的单细胞图像
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716274756193-8a2f8137-2f03-4b70-b4f0-94e086141c2d.png#averageHue=%23908f8f&clientId=u4e9dc7c5-4cce-4&from=paste&height=434&id=u11d51cf0&originHeight=651&originWidth=927&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=199559&status=done&style=none&taskId=ube82af48-6655-4729-b5f9-a9f638bfbb5&title=&width=618)
**这里的效果并不是很好，后期需要进行调整。尤其是细胞圆形过度。**
# 特征提取
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716994005832-427b9710-51e5-4ef4-bddb-d43edfd62ea5.png#averageHue=%23ece8df&clientId=uc035a44c-201b-4&from=paste&height=455&id=uf0a104f8&originHeight=682&originWidth=2098&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=255566&status=done&style=none&taskId=u6d27200a-71cb-444e-99ca-cf0bf199121&title=&width=1398.6666666666667)
## 数据测量Measure
### 对于FRET荧光数据需要测量的部分

- 查看荧光图像的纹理光谱，MeasureGranularity输出图像中纹理的尺寸测量的光谱。
- 查看AA、DD、DA图像荧光强度。
- 查看ED的图像荧光强度，表示的是AA、DD、DA的荧光强度关系。
- 查看相邻单细胞之间的关系，MeasureObjectNeighbors计算每个对象有多少个邻居，并记录关于邻居关系的各种属性，包括接触邻居的对象边缘像素的百分比。
- 查看荧光强度的分布情况，MeasureObjectIntensityDistribution 查看单细胞内荧光分布情况，其中参数 bin设置为4标识利用4个区间段区分荧光强度分布情况
- 查看单细胞形态特征，MeasureObjectSizeShape测量细胞大小等形态信息。
- 查看单细胞纹理特征，MeasureTexture测量图像和对象中纹理的程度和性质，以量化其粗糙度和平滑度。
### 对于明场图像需要测量的部分

- 输出明场细胞的中细胞区域的大小以及形态学特征
- 查看纹理特征，利用明场图像的单细胞内的区域变化进行查看
## 数据输出
输出两个CVS文件，一个是文件存放路径的metadata文件，一个是但细胞特征输出的文件。
# 分析模型
## 数据加载
### 数据清洗
删除无用的数据特征，如文件名称等，尤其是Metadata数据特征等

- **处理缺失值**：删除包含缺失值的行或列。
- **处理重复值**：删除完全重复的行或根据业务需求合并重复项。
- **处理异常值**：识别并处理异常值，如通过删除、替换、缩放到正常范围内或使用其他统计方法。
### 数据转换

- **标准化**：对于float64 以及 int64数据进行标准化操作。将数据按比例缩放，使其分布在均值为0、标准差为1的范围内。这通常用于数值型数据。
- **独热编码（One-Hot Encoding）**：将类别变量转换为机器学习算法可以理解的格式，通常用于处理分类数据。目前数据没有需要分类的特征。
- **标签编码（Label Encoding）**：将文本标签转换为整数。但这种方法可能导致模型误将标签视为有序数字，因此要小心使用。对于label需要进行标签编码操作。目前，处理的数据只有两个y值，一是加药实验组，二是对照组，分别采用了0，1进行标识。
### **特征选择**

- 从原始数据集中选择最有用的特征子集。这可以通过统计测试（如卡方检验、互信息）、模型权重、特征重要性评分或其他方法来实现。
- 本实验采用了**互信息**的方式实现了特征筛选，将筛选得出的特征使用了pickle进行存储。
## 数据降维UAMP算法
目前主流的三大数据降维算法，TSNE、PCA以及UMAP算法
## KNN分类算法

# 实验结果
目前数据集具有2h、4h、5h和6h，从6h开始实现简单的数据特征提取，以及降维分析操作。
## 仅通过FRET图像分析能否进行分类
### 验证6h的简单数据特征集中FRET图像提取的特征数据是否进行分类
未进行特征筛选的结果如下：
![result_FRET_NO_Filter_Feature_6h_UMAP.jpg](https://cdn.nlark.com/yuque/0/2024/jpeg/42860749/1716994795927-98225533-f412-49a8-b92e-0d55b0040187.jpeg#averageHue=%23f9f8f4&clientId=uc035a44c-201b-4&from=paste&height=320&id=ZKMSL&originHeight=480&originWidth=640&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=27571&status=done&style=none&taskId=u7d2cdd63-039b-4b36-a8a5-ea982e9ecec&title=&width=426.6666666666667)
**使用了AA、AD、DD、ED通道中提取强度特征以及单细胞大小和形态，两大类特征进行分析。**
特征筛选结果如下：原本115个特征筛选得到28个有效特征。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716974458769-0543c152-3171-4fe8-b1c1-590d437d78f7.png#averageHue=%23323130&clientId=u2e573a1d-007c-4&from=paste&height=632&id=u8cb7d4ef&originHeight=948&originWidth=1401&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=135477&status=done&style=none&taskId=u6fbb2e4b-1151-4a33-8311-82c2c03ad7a&title=&width=934)
筛选得到28个特征有用特征降维的结果如下：
![result_FRET_simple_6_UMAP.jpg](https://cdn.nlark.com/yuque/0/2024/jpeg/42860749/1716994848530-d9da00fa-7987-4f15-a765-27363a80e65e.jpeg#averageHue=%23f9f8ee&clientId=uc035a44c-201b-4&from=paste&height=320&id=ub0ed2268&originHeight=480&originWidth=640&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=37203&status=done&style=none&taskId=u0d463d2c-6aec-4670-a0e2-8c3d3e9102b&title=&width=426.6666666666667)
### 验证6h复杂特征数据信息能否进行细胞分类
利用上述数据测量的信息，提取了300多个单细胞特征。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716995071005-6f0e6a2a-3cb9-49f8-8a36-862022f7af16.png#averageHue=%23cab193&clientId=uc035a44c-201b-4&from=paste&height=113&id=u19407215&originHeight=169&originWidth=409&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=45830&status=done&style=none&taskId=ue2dcf9de-dfc2-4e7f-a713-54a0ea2e574&title=&width=272.6666666666667)
进而进行数据筛选，筛选得出149个特征进行分析，结果如下：
![result_FRET_6h_UMAP.jpg](https://cdn.nlark.com/yuque/0/2024/jpeg/42860749/1716995204869-85f16655-1888-4508-b349-b14394e50367.jpeg#averageHue=%23faf9f3&clientId=uc035a44c-201b-4&from=paste&height=320&id=u781969ae&originHeight=480&originWidth=640&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=26349&status=done&style=none&taskId=u8868c958-c624-4429-9bb9-f91ff41d080&title=&width=426.6666666666667)
### 验证2h复杂特征数据信息能够进行细胞分类
提取得出300多个复杂的单细胞特征，在未进行数据筛选的情况下，结果如下：
![result_FRET_NO_Filter_Feature_2h_UMAP.jpg](https://cdn.nlark.com/yuque/0/2024/jpeg/42860749/1716995252395-559ccc51-ff96-4f13-8a24-7f6c1d70a969.jpeg#averageHue=%23fafaf8&clientId=uc035a44c-201b-4&from=paste&height=320&id=u7749617f&originHeight=480&originWidth=640&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=25265&status=done&style=none&taskId=u911ff0d1-c31b-4c49-bc2a-c6bf4cf242b&title=&width=426.6666666666667)
在进行数据筛选情况下，依托于2h的特征数据，筛选得出54个有效特征，对应的结果如下：
![result_FRET_2h_UMAP.jpg](https://cdn.nlark.com/yuque/0/2024/jpeg/42860749/1716995306322-949530df-7406-4171-a969-427d7de54248.jpeg#averageHue=%23faf9f7&clientId=uc035a44c-201b-4&from=paste&height=320&id=u937678b1&originHeight=480&originWidth=640&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=31395&status=done&style=none&taskId=u3a3e9575-361b-4972-9279-9e09d1b6e65&title=&width=426.6666666666667)
而利用6h提取得出的特征149个进行2h特征的过滤，进行降维分析操作结果
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716995588397-1dcf1389-8f08-44d2-ae47-a7e00ae3435a.png#averageHue=%23fcfcfc&clientId=uc035a44c-201b-4&from=paste&height=662&id=u8bd87a8a&originHeight=993&originWidth=1314&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=85764&status=done&style=none&taskId=ua5a874c4-77b0-41e2-8197-fc0de1112bf&title=&width=876)
### 对比6h与2h有效特征之间的数量关系对比
6h有效特征为149个，而2h有效特征为54个，其中两者交集为21个。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/42860749/1716990337320-708f2c43-51ca-4096-8515-9d61469dfc74.png#averageHue=%232f3237&clientId=uc035a44c-201b-4&from=paste&height=791&id=u718ba8e1&originHeight=1186&originWidth=1392&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=257008&status=done&style=none&taskId=u48961ea6-8845-488a-af33-fc7042e6aa8&title=&width=928)
