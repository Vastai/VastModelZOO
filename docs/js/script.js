// 初始化 i18next
// 直接内联翻译资源，确保翻译功能正常工作
i18next.init({
    lng: 'zh',
    fallbackLng: 'zh',
    resources: {
        zh: {
            translation: {
                "loading": "加载中...",
                "subtitle": "VastModelZOO提供人工智能多个领域（CV/NLP/LLM/MLLM等）的开源模型在瀚博GPU芯片上的部署示例",
                "totalModels": "总模型数量",
                "modelCategories": "模型分类（大类）",
                "modelSubcategories": "模型分类（子类）",
                "inferenceFrameworks": "推理框架",
                "searchPlaceholder": "搜索模型名称、类型或推理框架...",
                "modelClassification": "模型分类",
                "computerVision": "计算机视觉模型",
                "naturalLanguageProcessing": "自然语言处理模型",
                "largeLanguageModel": "大语言模型",
                "visionLanguageModel": "视觉语言模型",
                "audioLanguageModel": "语音语言模型",
                "imageClassification": "图像分类",
                "objectDetection": "目标检测",
                "objectDetection3d": "3D目标检测",
                "imageSegmentation": "图像分割",
                "faceAlignment": "人脸对齐",
                "faceDetection": "人脸检测",
                "faceQuality": "人脸质量评估",
                "faceRecognition": "人脸识别",
                "facialAttribute": "人脸属性分析",
                "imageColorization": "图像着色",
                "imageCompression": "图像压缩",
                "imageRetrieval": "图像检索",
                "lowLightImageEnhancement": "低光照图像增强",
                "multiObjectTracking": "多目标跟踪",
                "poseEstimation": "姿态估计",
                "reidentification": "重识别",
                "salientObjectDetection": "显著性目标检测",
                "superResolution": "超分辨率",
                "textDetection": "文本检测",
                "textRecognition": "文本识别",
                "informationExtraction": "信息提取",
                "namedEntityRecognition": "命名实体识别",
                "questionAnswering": "问答系统",
                "sentenceClassification": "句子分类",
                "textVectorization": "文本向量化",
                "reranker": "重排序",
                "generalLLM": "通用大语言模型",
                "domainLLM": "领域大语言模型",
                "generalVLM": "通用视觉语言模型",
                "domainVLM": "领域视觉语言模型",
                "generalALM": "通用语音语言模型",
                "domainALM": "领域语音语言模型",
                "viewModels": "查看模型",
                "cvModels": "CV Models - 计算机视觉模型",
                "nlpModels": "NLP Models - 自然语言处理模型",
                "llmModels": "LLM Models - 大语言模型",
                "vlmModels": "VLM Models - 视觉语言模型",
                "almModels": "ALM Models - 语音语言模型",
                "noModelsFound": "没有找到以 {{letter}} 开头的模型",
                "searchResults": "共搜索 {{count}} 个结果",
                "searchResultsWithDetails": "模型名称共找到 {{modelCount}} 个结果，模型列表共找到 {{listCount}} 个结果",
                "language": "语言",
                "chinese": "中文",
                "english": "English",
                "modelName": "模型名称",
                "modelSource": "模型来源",
                "modelList": "模型列表",
                "modelType": "模型类型",
                "runtime": "运行时",
                "clearSearch": "清除搜索",
                "searchResultsCount": "搜索结果统计",
                "vastaiWebsite": "瀚博官网",
                "pageUpdating": "© 2023-2026 VastModelZOO | 页面持续更新，与GitHub仓库模型列表同步",
                "backToTop": "返回顶部",
                "vastaiInferenceFramework": "瀚博自研推理框架",
                "adaptedToVllm": "适配vLLM推理框架",
                "performanceAccuracy": "性能/精度",
                "comprehensiveEvaluation": "全面的模型推理、性能和精度评估流程"
            }
        },
        en: {
            translation: {
                "loading": "Loading...",
                "subtitle": "VastModelZOO provides deployment examples of open-source models across multiple AI domains (CV/NLP/LLM/MLLM, etc.) on Vastai GPU chips",
                "totalModels": "Total Number of Models",
                "modelCategories": "Model Classification (Major Types)",
                "modelSubcategories": "Model Classification (Subtypes)",
                "inferenceFrameworks": "Inference Frameworks",
                "searchPlaceholder": "Search by model name, type, or inference framework...",
                "modelClassification": "Model Classification",
                "computerVision": "CV Models",
                "naturalLanguageProcessing": "NLP Models",
                "largeLanguageModel": "LLM Models",
                "visionLanguageModel": "VLM Models",
                "audioLanguageModel": "ALM Models",
                "imageClassification": "Image Classification",
                "objectDetection": "Object Detection",
                "objectDetection3d": "3D Object Detection",
                "imageSegmentation": "Image Segmentation",
                "faceAlignment": "Face Alignment",
                "faceDetection": "Face Detection",
                "faceQuality": "Face Quality Assessment",
                "faceRecognition": "Face Recognition",
                "facialAttribute": "Facial Attribute Analysis",
                "imageColorization": "Image Colorization",
                "imageCompression": "Image Compression",
                "imageRetrieval": "Image Retrieval",
                "lowLightImageEnhancement": "Low-Light Image Enhancement",
                "multiObjectTracking": "Multi-Object Tracking",
                "poseEstimation": "Pose Estimation",
                "reidentification": "Re-Identification",
                "salientObjectDetection": "Salient Object Detection",
                "superResolution": "Super Resolution",
                "textDetection": "Text Detection",
                "textRecognition": "Text Recognition",
                "informationExtraction": "Information Extraction",
                "namedEntityRecognition": "Named Entity Recognition",
                "questionAnswering": "Question Answering",
                "sentenceClassification": "Sentence Classification",
                "textVectorization": "Embedding",
                "reranker": "Reranker",
                "generalLLM": "General LLM Models",
                "domainLLM": "Domain LLM Models",
                "generalVLM": "General VLM Models",
                "domainVLM": "Domain VLM Models",
                "generalALM": "General ALM Models",
                "domainALM": "Domain ALM Models",
                "viewModels": "View Models",
                "cvModels": "CV Models - Computer Vision Models",
                "nlpModels": "NLP Models - Natural Language Processing Models",
                "llmModels": "LLM Models - Large Language Models",
                "vlmModels": "VLM Models - Vision Language Models",
                "almModels": "ALM Models - Audio Language Models",
                "noModelsFound": "No models found starting with {{letter}}",
                "searchResults": "{{count}} result(s) found",
                "searchResultsWithDetails": "{{modelCount}} result(s) found by model name, {{listCount}} result(s) found in model list",
                "language": "Language",
                "chinese": "中文",
                "english": "English",
                "modelName": "Model Name",
                "modelSource": "Model Source",
                "modelList": "Model List",
                "modelType": "Model Type",
                "runtime": "Runtime",
                "clearSearch": "Clear Search",
                "searchResultsCount": "Search Results",
                "vastaiWebsite": "Vastai Website",
                "pageUpdating": "© 2023-2026 VastModelZOO |Page continuously updated and synchronized with the GitHub repository model list",
                "backToTop": "Back to top",
                "vastaiInferenceFramework": "Vastai Proprietary Inference Framework",
                "adaptedToVllm": "Adapted to vLLM Inference Framework",
                "performanceAccuracy": "Performance/Accuracy",
                "comprehensiveEvaluation": "Comprehensive model inference, performance and accuracy evaluation process"
            }
        }
    }
}, function(err, t) {
    if (err) console.error('i18next initialization error:', err);
    console.log('i18next initialized successfully');
    console.log('Current language:', i18next.language);
    console.log('Translation test on init:', i18next.t('loading'));
    updateContent();
});

// 缓存DOM元素，避免重复查询
const domElements = {
    loaderText: document.querySelector('.loader-text'),
    subtitle: document.querySelector('.subtitle-main'),
    statLabels: document.querySelectorAll('.stat-label'),
    searchInput: document.getElementById('model-search'),
    modelClassification: document.querySelector('.sidebar-section h3'),
    categoryLinks: document.querySelectorAll('.category-list > li > a'),
    subcategoryLinks: document.querySelectorAll('.subcategory-list a'),
    categoryHeaders: document.querySelectorAll('.category-header h2'),
    subcategoryHeaders: document.querySelectorAll('.subcategory-header h3')
};

// 子分类键值映射（一次定义，多次使用）
const subcategoryKeys = {
    '图像分类': 'imageClassification',
    '目标检测': 'objectDetection',
    '3D目标检测': 'objectDetection3d',
    '图像分割': 'imageSegmentation',
    '人脸对齐': 'faceAlignment',
    '人脸检测': 'faceDetection',
    '人脸质量评估': 'faceQuality',
    '人脸识别': 'faceRecognition',
    '人脸属性分析': 'facialAttribute',
    '图像着色': 'imageColorization',
    '图像压缩': 'imageCompression',
    '图像检索': 'imageRetrieval',
    '低光照图像增强': 'lowLightImageEnhancement',
    '多目标跟踪': 'multiObjectTracking',
    '姿态估计': 'poseEstimation',
    '重识别': 'reidentification',
    '显著性目标检测': 'salientObjectDetection',
    '超分辨率': 'superResolution',
    '文本检测': 'textDetection',
    '文本识别': 'textRecognition',
    '信息提取': 'informationExtraction',
    '命名实体识别': 'namedEntityRecognition',
    '问答系统': 'questionAnswering',
    '句子分类': 'sentenceClassification',
    '文本向量化': 'textVectorization',
    '重排序': 'reranker',
    '通用大语言模型': 'generalLLM',
    '领域大语言模型': 'domainLLM',
    '通用视觉语言模型': 'generalVLM',
    '领域视觉语言模型': 'domainVLM',
    '通用语音语言模型': 'generalALM',
    '领域语音语言模型': 'domainALM'
};

// 为子分类链接设置data-key属性（只执行一次）
function setupSubcategoryKeys() {
    domElements.subcategoryLinks.forEach(link => {
        const originalText = link.textContent.trim();
        if (subcategoryKeys[originalText]) {
            link.setAttribute('data-key', subcategoryKeys[originalText]);
        }
    });
}

// 初始化时设置一次
setupSubcategoryKeys();

// 更新页面内容为当前语言
function updateContent() {
    // 更新加载动画文本
    const loaderText = document.querySelector('.loader-text');
    if (loaderText) loaderText.textContent = i18next.t('loading');
    
    // 更新副标题
    const subtitle = document.querySelector('.subtitle-main');
    if (subtitle) subtitle.textContent = i18next.t('subtitle');
    
    // 更新统计标签
    const statLabels = document.querySelectorAll('.stat-label');
    if (statLabels.length >= 4) {
        statLabels[0].textContent = i18next.t('totalModels');
        statLabels[1].textContent = i18next.t('modelCategories');
        statLabels[2].textContent = i18next.t('modelSubcategories');
        statLabels[3].textContent = i18next.t('inferenceFrameworks');
    }
    
    // 更新搜索框占位符
    const searchInput = document.getElementById('model-search');
    if (searchInput) searchInput.placeholder = i18next.t('searchPlaceholder');
    
    // 更新模型分类标题
    const modelClassification = document.querySelector('.sidebar-section h3');
    if (modelClassification) modelClassification.innerHTML = `<i class="fas fa-list"></i> ${i18next.t('modelClassification')}`;
    
    // 更新分类链接
    const categoryLinks = document.querySelectorAll('.category-list > li > a');
    if (categoryLinks.length >= 4) {
        categoryLinks[0].innerHTML = `<i class="fas fa-eye"></i> ${i18next.t('computerVision')}`;
        categoryLinks[1].innerHTML = `<i class="fas fa-language"></i> ${i18next.t('naturalLanguageProcessing')}`;
        categoryLinks[2].innerHTML = `<i class="fas fa-robot"></i> ${i18next.t('largeLanguageModel')}`;
        categoryLinks[3].innerHTML = `<i class="fas fa-images"></i> ${i18next.t('visionLanguageModel')}`;
        categoryLinks[4].innerHTML = `<i class="fas fa-file-audio"></i> ${i18next.t('audioLanguageModel')}`;
    }
    
    // 更新子分类链接
    const subcategoryLinks = document.querySelectorAll('.subcategory-list a');
    const subcategoryKeys = {
        '图像分类': 'imageClassification',
        '目标检测': 'objectDetection',
        '3D目标检测': 'objectDetection3d',
        '图像分割': 'imageSegmentation',
        '人脸对齐': 'faceAlignment',
        '人脸检测': 'faceDetection',
        '人脸质量评估': 'faceQuality',
        '人脸识别': 'faceRecognition',
        '人脸属性分析': 'facialAttribute',
        '图像着色': 'imageColorization',
        '图像压缩': 'imageCompression',
        '图像检索': 'imageRetrieval',
        '低光照图像增强': 'lowLightImageEnhancement',
        '多目标跟踪': 'multiObjectTracking',
        '姿态估计': 'poseEstimation',
        '重识别': 'reidentification',
        '显著性目标检测': 'salientObjectDetection',
        '超分辨率': 'superResolution',
        '文本检测': 'textDetection',
        '文本识别': 'textRecognition',
        '信息提取': 'informationExtraction',
        '命名实体识别': 'namedEntityRecognition',
        '问答系统': 'questionAnswering',
        '句子分类': 'sentenceClassification',
        '文本向量化': 'textVectorization',
        '重排序': 'reranker',
        '通用大语言模型': 'generalLLM',
        '领域大语言模型': 'domainLLM',
        '通用视觉语言模型': 'generalVLM',
        '领域视觉语言模型': 'domainVLM',
        '通用语音语言模型': 'generalALM',
        '领域语音语言模型': 'domainALM'
    };
    
    // 为每个子分类链接设置data-key属性，用于后续翻译
    subcategoryLinks.forEach(link => {
        const originalText = link.textContent.trim();
        if (subcategoryKeys[originalText]) {
            link.setAttribute('data-key', subcategoryKeys[originalText]);
        }
    });
    
    // 根据当前语言更新子分类链接文本
    subcategoryLinks.forEach(link => {
        const key = link.getAttribute('data-key');
        if (key) {
            link.textContent = i18next.t(key);
        }
    });
    
    // 更新类别标题
    const categoryHeaders = document.querySelectorAll('.category-header h2');
    if (categoryHeaders.length >= 4) {
        categoryHeaders[0].innerHTML = `<i class="fas fa-eye"></i> ${i18next.t('cvModels')}`;
        categoryHeaders[1].innerHTML = `<i class="fas fa-language"></i> ${i18next.t('nlpModels')}`;
        categoryHeaders[2].innerHTML = `<i class="fas fa-robot"></i> ${i18next.t('llmModels')}`;
        categoryHeaders[3].innerHTML = `<i class="fas fa-images"></i> ${i18next.t('vlmModels')}`;
        categoryHeaders[4].innerHTML = `<i class="fas fa-file-audio"></i> ${i18next.t('almModels')}`;
    }
    
    // 更新子类别标题
    const subcategoryHeaders = document.querySelectorAll('.subcategory-header h3');
    subcategoryHeaders.forEach((header, index) => {
        const icon = header.querySelector('i');
        if (icon) {
            // 为每个子类别标题设置固定的翻译键
            const sectionId = header.closest('.subcategory-item').id;
            let translationKey = null;
            
            // 根据sectionId映射到对应的翻译键
            const sectionToKey = {
                'cv-classification': 'imageClassification',
                'cv-detection': 'objectDetection',
                'cv-detection3d': 'objectDetection3d',
                'cv-segmentation': 'imageSegmentation',
                'cv-face-alignment': 'faceAlignment',
                'cv-face-detection': 'faceDetection',
                'cv-face-quality': 'faceQuality',
                'cv-face-recognize': 'faceRecognition',
                'cv-facial-attribute': 'facialAttribute',
                'cv-image-colorization': 'imageColorization',
                'cv-image-compress': 'imageCompression',
                'cv-image-retrieval': 'imageRetrieval',
                'cv-low-light-image-enhancement': 'lowLightImageEnhancement',
                'cv-mot': 'multiObjectTracking',
                'cv-pose': 'poseEstimation',
                'cv-reid': 'reidentification',
                'cv-salient-object-detection': 'salientObjectDetection',
                'cv-super-resolution': 'superResolution',
                'cv-text-detection': 'textDetection',
                'cv-text-recognition': 'textRecognition',
                'nlp-information-extraction': 'informationExtraction',
                'nlp-named-entity-recognition': 'namedEntityRecognition',
                'nlp-question-answering': 'questionAnswering',
                'nlp-sentence-classification': 'sentenceClassification',
                'nlp-text2vec': 'textVectorization',
                'nlp-reranker': 'reranker',
                'llm-general': 'generalLLM',
                'llm-domain': 'domainLLM',
                'vlm-general': 'generalVLM',
                'vlm-domain': 'domainVLM',
                'alm-general': 'generalALM',
                'alm-domain': 'domainALM'
            };
            
            if (sectionToKey[sectionId]) {
                translationKey = sectionToKey[sectionId];
            }
            
            if (translationKey) {
                header.innerHTML = `<i class="${icon.className}"></i> ${i18next.t(translationKey)}`;
            }
        }
    });
    
    // 更新查看模型按钮
    const viewModelButtons = document.querySelectorAll('.toggle-details');
    viewModelButtons.forEach(button => {
        button.innerHTML = `<i class="fas fa-list"></i> ${i18next.t('viewModels')}`;
    });
    
    // 更新表格表头
    const tables = document.querySelectorAll('.models-table');
    tables.forEach(table => {
        const tableHeaders = table.querySelectorAll('th');
        if (tableHeaders.length >= 5) {
            tableHeaders[0].textContent = i18next.t('modelName');
            tableHeaders[1].textContent = i18next.t('modelSource');
            tableHeaders[2].textContent = i18next.t('modelList');
            tableHeaders[3].textContent = i18next.t('modelType');
            tableHeaders[4].textContent = i18next.t('runtime');
        }
    });
    
    // 更新清除搜索按钮
    const clearSearchBtn = document.querySelector('.clear-search');
    if (clearSearchBtn) {
        clearSearchBtn.setAttribute('title', i18next.t('clearSearch'));
    }
    
    // 更新搜索结果计数
    const searchCount = document.getElementById('search-count');
    if (searchCount) {
        searchCount.setAttribute('title', i18next.t('searchResultsCount'));
        // 如果搜索结果计数可见，重新调用searchModels函数更新文本内容
        // 这样可以利用现有的正确逻辑来计算和显示搜索结果
        const searchInput = document.getElementById('model-search');
        const searchTerm = searchInput.value.toLowerCase().trim();
        if (searchTerm) {
            // 重新调用searchModels函数，利用其正确的计算逻辑
            searchModels();
        }
    }
    
    // 更新footer部分
    const footerLinks = document.querySelectorAll('.footer-links a');
    if (footerLinks.length >= 4) {
        footerLinks[0].innerHTML = `<i class="fas fa-globe"></i> ${i18next.t('vastaiWebsite')}`;
    }
    
    const footerCopyright = document.querySelector('.footer-copyright p:first-child');
    if (footerCopyright) {
        // 直接设置文本内容，确保翻译生效
        footerCopyright.textContent = i18next.t('pageUpdating');
    }
    
    // 更新返回顶部按钮
    const backToTopBtn = document.getElementById('backToTop');
    if (backToTopBtn) {
        backToTopBtn.title = i18next.t('backToTop');
    }
    
    // 更新三个功能卡片
    const featureCards = document.querySelectorAll('div[style*="grid-template-columns"] > div');
    if (featureCards.length >= 3) {
        // 第一个卡片：Build_In
        const buildInCard = featureCards[0];
        const buildInDesc = buildInCard.querySelector('p');
        if (buildInDesc) {
            buildInDesc.textContent = i18next.t('vastaiInferenceFramework');
        }
        
        // 第二个卡片：vLLM
        const vllmCard = featureCards[1];
        const vllmDesc = vllmCard.querySelector('p');
        if (vllmDesc) {
            vllmDesc.textContent = i18next.t('adaptedToVllm');
        }
        
        // 第三个卡片：性能/精度
        const perfCard = featureCards[2];
        const perfTitle = perfCard.querySelector('h5');
        const perfDesc = perfCard.querySelector('p');
        if (perfTitle && perfDesc) {
            perfTitle.textContent = i18next.t('performanceAccuracy');
            perfDesc.textContent = i18next.t('comprehensiveEvaluation');
        }
    }
    
    // 更新footer中的瀚博官网
    const vastaiLink = document.querySelector('.footer-links a:first-child');
    if (vastaiLink) {
        const currentLang = i18next.language;
        if (currentLang === 'en') {
            vastaiLink.innerHTML = `<i class="fas fa-globe"></i> ${i18next.t('vastaiWebsite')}`;
            vastaiLink.href = 'https://www.vastai.com/en';
        } else {
            vastaiLink.innerHTML = `<i class="fas fa-globe"></i> 瀚博官网`;
            vastaiLink.href = 'https://www.vastai.com';
        }
    }
    
    // 更新header中的瀚博官网链接
    const headerVastaiLink = document.getElementById('vastai-website-link');
    if (headerVastaiLink) {
        const currentLang = i18next.language;
        if (currentLang === 'en') {
            headerVastaiLink.href = 'https://www.vastai.com/en';
        } else {
            headerVastaiLink.href = 'https://www.vastai.com';
        }
    }
    
    // 更新ModelScope链接
    const modelscopeLink = document.getElementById('modelscope-link');
    if (modelscopeLink) {
        const currentLang = i18next.language;
        if (currentLang === 'en') {
            modelscopeLink.href = 'https://www.modelscope.cn/home';
            modelscopeLink.title = 'Visit ModelScope Website';
        } else {
            modelscopeLink.href = 'https://www.modelscope.cn/home';
            modelscopeLink.title = '访问ModelScope官网';
        }
    }
}

// 切换语言
function changeLanguage(lang) {
    console.log('Changing language to:', lang);
    i18next.changeLanguage(lang, function(err, t) {
        if (err) {
            console.error('Language change error:', err);
        } else {
            console.log('Language changed successfully to:', i18next.language);
            console.log('Translation test:', i18next.t('loading'));
            console.log('Translation test 2:', i18next.t('subtitle'));
            // 确保翻译文件加载完成后再更新页面内容
            setTimeout(updateContent, 200); // 增加延迟确保翻译文件加载完成
        }
        
        // 更新语言按钮的active状态
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`btn-${lang}`).classList.add('active');
    });
}

// 页面加载完成后隐藏加载动画
window.addEventListener('load', function() {
    const pageLoader = document.getElementById('pageLoader');
    if (pageLoader) {
        setTimeout(() => {
            pageLoader.classList.add('hidden');
            document.body.classList.remove('loading');
        }, 300);
    }
    
    // 初始化语言按钮的active状态
    setTimeout(() => {
        const currentLang = i18next.language || 'zh';
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        const langBtn = document.getElementById(`btn-${currentLang}`);
        if (langBtn) {
            langBtn.classList.add('active');
        } else {
            document.getElementById('btn-zh').classList.add('active');
        }
    }, 100);
});

// 切换大类内容显示/隐藏
function toggleCategory(categoryId) {
    console.log('toggleCategory called with:', categoryId);
    const content = document.getElementById(categoryId);
    const header = content.previousElementSibling;
    
    if (!content) {
        console.error('Content element not found for categoryId:', categoryId);
        return;
    }
    
    console.log('Current content classes:', content.className);
    console.log('Current header classes:', header ? header.className : 'no header');
    
    const navIdMap = {
        'cv-content': 'cv-sub',
        'nlp-content': 'nlp-sub',
        'llm-content': 'llm-sub',
        'vlm-content': 'vlm-sub',
        'alm-content': 'alm-sub'
    };
    const categoryLinkMap = {
        'cv-content': '#cv-models',
        'nlp-content': '#nlp-models',
        'llm-content': '#llm-models',
        'vlm-content': '#vlm-models',
        'alm-content': '#alm-models'
    };
    
    const isExpanding = !content.classList.contains('expanded');
    console.log('isExpanding:', isExpanding);
    
    if (isExpanding) {
        // 清除所有子分类链接的高亮
        document.querySelectorAll('.subcategory-list a').forEach(link => {
            link.classList.remove('active');
        });
        
        // 折叠所有其他大类
        document.querySelectorAll('.subcategory-panel').forEach(panel => {
            if (panel.id !== categoryId) {
                panel.classList.remove('expanded');
                const panelHeader = panel.previousElementSibling;
                if (panelHeader) {
                    panelHeader.classList.remove('active');
                }
                
                // 折叠该大类内的所有子类别
                panel.querySelectorAll('.models-table-container').forEach(container => {
                    container.classList.remove('expanded');
                    const containerHeader = container.previousElementSibling;
                    if (containerHeader) {
                        containerHeader.classList.remove('active');
                    }
                    // 关闭该子类别内的所有模型列表
                    container.querySelectorAll('.model-list-details').forEach(el => {
                        el.style.display = 'none';
                    });
                });
            }
        });
        
        // 关闭当前大类内的所有模型列表（确保展开时模型列表是折叠状态）
        content.querySelectorAll('.model-list-details').forEach(el => {
            el.style.display = 'none';
        });
        
        // 同步折叠左侧导航栏的所有子分类列表
        document.querySelectorAll('.subcategory-list').forEach(list => {
            list.classList.remove('expanded');
            const listButton = list.previousElementSibling;
            if (listButton && listButton.classList.contains('toggle-subcategories')) {
                const listIcon = listButton.querySelector('i');
                if (listIcon) {
                    listIcon.className = 'fas fa-chevron-down';
                }
            }
        });
        
        // 展开左侧导航栏对应的子分类列表
        const targetNavId = navIdMap[categoryId];
        const targetLink = categoryLinkMap[categoryId];
        if (targetNavId) {
            const targetNavList = document.getElementById(targetNavId);
            if (targetNavList) {
                targetNavList.classList.add('expanded');
                const navButton = targetNavList.previousElementSibling;
                if (navButton && navButton.classList.contains('toggle-subcategories')) {
                    const navIcon = navButton.querySelector('i');
                    if (navIcon) {
                        navIcon.className = 'fas fa-chevron-up';
                    }
                }
            }
        }
        // 高亮左侧导航栏对应的类别链接
        if (targetLink) {
            document.querySelectorAll('.category-list > li > a').forEach(link => {
                link.classList.remove('active');
            });
            const targetLinkElement = document.querySelector(`.category-list > li > a[href="${targetLink}"]`);
            if (targetLinkElement) {
                targetLinkElement.classList.add('active');
            }
        }
    } else {
        console.log('Entering else branch (collapsing)');
        // 折叠当前大类时,同时折叠其内部的所有子类别
        content.querySelectorAll('.models-table-container').forEach(container => {
            container.classList.remove('expanded');
            const containerHeader = container.previousElementSibling;
            if (containerHeader) {
                containerHeader.classList.remove('active');
            }
            // 关闭该子类别内的所有模型列表
            container.querySelectorAll('.model-list-details').forEach(el => {
                el.style.display = 'none';
            });
        });
        
        // 同步折叠左侧导航栏对应的子分类列表
        const targetNavId = navIdMap[categoryId];
        console.log('targetNavId:', targetNavId);
        if (targetNavId) {
            const targetNavList = document.getElementById(targetNavId);
            console.log('targetNavList:', targetNavList);
            if (targetNavList) {
                targetNavList.classList.remove('expanded');
                console.log('Removed expanded from navList');
                const navButton = targetNavList.previousElementSibling;
                if (navButton && navButton.classList.contains('toggle-subcategories')) {
                    const navIcon = navButton.querySelector('i');
                    if (navIcon) {
                        navIcon.className = 'fas fa-chevron-down';
                    }
                }
                // 移除该列表下所有子类别链接的高亮
                targetNavList.querySelectorAll('a').forEach(link => {
                    link.classList.remove('active');
                });
            }
        }
        
        // 移除左侧导航栏对应类别的高亮
        const targetLink = categoryLinkMap[categoryId];
        console.log('targetLink:', targetLink);
        if (targetLink) {
            const targetLinkElement = document.querySelector(`.category-list > li > a[href="${targetLink}"]`);
            console.log('targetLinkElement:', targetLinkElement);
            if (targetLinkElement) {
                targetLinkElement.classList.remove('active');
            }
        }
    }

    console.log('Before final class manipulation, isExpanding:', isExpanding);
    if (isExpanding) {
        content.classList.add('expanded');
        header.classList.add('active');
        console.log('Added expanded and active classes');
    } else {
        content.classList.remove('expanded');
        header.classList.remove('active');
        console.log('Removed expanded and active classes');
    }
    console.log('After manipulation, content classes:', content.className);
    console.log('After manipulation, header classes:', header ? header.className : 'no header');
    
    // 如果是展开操作，滚动到该类别的正确位置
    if (isExpanding) {
        const categoryCard = content.closest('.category-card');
        if (categoryCard && window.categoryPositions && window.categoryPositions.categories) {
            const categoryId = categoryCard.id;
            if (window.categoryPositions.categories[categoryId]) {
                setTimeout(() => {
                    window.scrollTo({
                        top: window.categoryPositions.categories[categoryId] - 120,
                        behavior: 'smooth'
                    });
                }, 100);
            }
        }
    }
}

// 切换子类别内容显示/隐藏
function toggleSubcategory(subcategoryId) {
    const content = document.getElementById(subcategoryId);
    const header = content.previousElementSibling;
    
    const isExpanding = !content.classList.contains('expanded');
    
    if (isExpanding) {
        // 折叠同一大类下的所有其他子类别
        const parentPanel = content.closest('.subcategory-panel');
        if (parentPanel) {
            parentPanel.querySelectorAll('.models-table-container').forEach(container => {
                if (container.id !== subcategoryId) {
                    container.classList.remove('expanded');
                    const containerHeader = container.previousElementSibling;
                    if (containerHeader) {
                        containerHeader.classList.remove('active');
                    }
                }
            });
        }
        
        // 同步折叠左侧导航栏的所有子分类列表(除了当前大类对应的)
        const currentCategoryCard = content.closest('.category-card');
        let currentCategoryId = null;
        if (currentCategoryCard) {
            const categoryIdMap = {
                'cv-models': 'cv-sub',
                'nlp-models': 'nlp-sub',
                'llm-models': 'llm-sub',
                'vlm-models': 'vlm-sub',
                'alm-models': 'alm-sub'
            };
            currentCategoryId = categoryIdMap[currentCategoryCard.id];
        }
        
        document.querySelectorAll('.subcategory-list').forEach(list => {
            if (list.id !== currentCategoryId) {
                list.classList.remove('expanded');
                const listButton = list.previousElementSibling;
                if (listButton && listButton.classList.contains('toggle-subcategories')) {
                    const listIcon = listButton.querySelector('i');
                    if (listIcon) {
                        listIcon.className = 'fas fa-chevron-down';
                    }
                }
            }
        });
        
        // 关闭所有其他子类别中已展开的模型列表
        if (parentPanel) {
            parentPanel.querySelectorAll('.model-list-details').forEach(el => {
                const elContainer = el.closest('.models-table-container');
                if (elContainer && elContainer.id !== subcategoryId) {
                    el.style.display = 'none';
                }
            });
        }
        
        // 高亮左侧导航栏对应的子类别链接
        const targetSubLink = `#${subcategoryId.replace('-content', '')}`;
        document.querySelectorAll('.subcategory-list a').forEach(link => {
            link.classList.remove('active');
        });
        const targetSubLinkElement = document.querySelector(`.subcategory-list a[href="${targetSubLink}"]`);
        if (targetSubLinkElement) {
            targetSubLinkElement.classList.add('active');
        }
    } else {
        // 折叠当前子类别时，关闭其内的所有模型列表
        content.querySelectorAll('.model-list-details').forEach(el => {
            el.style.display = 'none';
        });
        
        // 移除左侧导航栏对应子类别的高亮
        const targetSubLink = `#${subcategoryId.replace('-content', '')}`;
        const targetSubLinkElement = document.querySelector(`.subcategory-list a[href="${targetSubLink}"]`);
        if (targetSubLinkElement) {
            targetSubLinkElement.classList.remove('active');
        }
    }

    content.classList.toggle('expanded');
    header.classList.toggle('active');

    // 添加动画效果
    content.classList.add('fade-in');
    setTimeout(() => {
        content.classList.remove('fade-in');
    }, 300);
    
    // 如果是展开操作，滚动到该子类别的正确位置
    if (isExpanding) {
        const subcategoryItem = content.closest('.subcategory-item');
        if (subcategoryItem && window.categoryPositions && window.categoryPositions.subcategories) {
            const subcategoryId = subcategoryItem.id;
            if (window.categoryPositions.subcategories[subcategoryId]) {
                setTimeout(() => {
                    window.scrollTo({
                        top: window.categoryPositions.subcategories[subcategoryId] - 120,
                        behavior: 'smooth'
                    });
                }, 150);
            }
        }
    }
}

// 切换侧边栏子类别显示/隐藏
function toggleSubcategories(subId) {
    const sublist = document.getElementById(subId);
    const button = sublist.previousElementSibling;
    
    const isExpanding = !sublist.classList.contains('expanded');
    
    // 左侧导航栏子分类列表ID与主页面大类内容ID的映射
    const navToContentMap = {
        'cv-sub': 'cv-content',
        'nlp-sub': 'nlp-content',
        'llm-sub': 'llm-content',
        'vlm-sub': 'vlm-content',
        'alm-sub': 'alm-content'
    };
    const subToCategoryLinkMap = {
        'cv-sub': '#cv-models',
        'nlp-sub': '#nlp-models',
        'llm-sub': '#llm-models',
        'vlm-sub': '#vlm-models',
        'alm-sub': '#alm-models'
    };
    
    if (isExpanding) {
        // 折叠所有其他子分类列表
        document.querySelectorAll('.subcategory-list').forEach(list => {
            if (list.id !== subId) {
                list.classList.remove('expanded');
                // 更新对应按钮的图标
                const listButton = list.previousElementSibling;
                if (listButton && listButton.classList.contains('toggle-subcategories')) {
                    const listIcon = listButton.querySelector('i');
                    if (listIcon) {
                        listIcon.className = 'fas fa-chevron-down';
                    }
                }
                // 移除该列表下所有子类别链接的高亮
                list.querySelectorAll('a').forEach(link => {
                    link.classList.remove('active');
                });
                
                // 同步折叠主页面对应的大类
                const contentId = navToContentMap[list.id];
                if (contentId) {
                    const content = document.getElementById(contentId);
                    if (content) {
                        content.classList.remove('expanded');
                        const header = content.previousElementSibling;
                        if (header) {
                            header.classList.remove('active');
                        }
                        // 折叠该大类内的所有子类别
                        content.querySelectorAll('.models-table-container').forEach(container => {
                            container.classList.remove('expanded');
                            const containerHeader = container.previousElementSibling;
                            if (containerHeader) {
                                containerHeader.classList.remove('active');
                            }
                        });
                    }
                }
            }
        });
        
        // 同步展开主页面对应的大类
        const contentId = navToContentMap[subId];
        if (contentId) {
            const content = document.getElementById(contentId);
            if (content) {
                content.classList.add('expanded');
                const header = content.previousElementSibling;
                if (header) {
                    header.classList.add('active');
                }
            }
        }
        
        // 高亮左侧导航栏对应的类别链接
        const targetLink = subToCategoryLinkMap[subId];
        if (targetLink) {
            document.querySelectorAll('.category-list > li > a').forEach(link => {
                link.classList.remove('active');
            });
            const targetLinkElement = document.querySelector(`.category-list > li > a[href="${targetLink}"]`);
            if (targetLinkElement) {
                targetLinkElement.classList.add('active');
            }
        }
        
        // 滚动到对应大类的位置，与点击文字链接的行为保持一致
        const targetId = targetLink ? targetLink.substring(1) : null;
        if (targetId) {
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                // 使用预先计算的位置信息进行滚动
                const targetTop = window.categoryPositions && window.categoryPositions.categories[targetId] 
                    ? window.categoryPositions.categories[targetId] - 120 
                    : targetElement.offsetTop - 120;
                
                setTimeout(() => {
                    window.scrollTo({
                        top: targetTop,
                        behavior: 'smooth'
                    });
                }, 100);
            }
        }
    } else {
        // 折叠时移除当前列表下所有子类别链接的高亮
        sublist.querySelectorAll('a').forEach(link => {
            link.classList.remove('active');
        });
        
        // 移除左侧导航栏对应的类别链接高亮
        const targetLink = subToCategoryLinkMap[subId];
        if (targetLink) {
            const targetLinkElement = document.querySelector(`.category-list > li > a[href="${targetLink}"]`);
            if (targetLinkElement) {
                targetLinkElement.classList.remove('active');
            }
        }
        
        // 同步折叠主页面对应的大类
        const contentId = navToContentMap[subId];
        if (contentId) {
            const content = document.getElementById(contentId);
            if (content) {
                content.classList.remove('expanded');
                const header = content.previousElementSibling;
                if (header) {
                    header.classList.remove('active');
                }
                // 折叠该大类内的所有子类别
                content.querySelectorAll('.models-table-container').forEach(container => {
                    container.classList.remove('expanded');
                    const containerHeader = container.previousElementSibling;
                    if (containerHeader) {
                        containerHeader.classList.remove('active');
                    }
                });
            }
        }
    }

    sublist.classList.toggle('expanded');
    const icon = button.querySelector('i');
    if (sublist.classList.contains('expanded')) {
        icon.className = 'fas fa-chevron-up';
    } else {
        icon.className = 'fas fa-chevron-down';
    }
}

// 切换模型列表显示/隐藏
function toggleModelList(modelId) {
    const details = document.getElementById(modelId);
    if (details.style.display === 'block') {
        details.style.display = 'none';
    } else {
        // 隐藏其他打开的详情
        document.querySelectorAll('.model-list-details').forEach(el => {
            if (el.id !== modelId) el.style.display = 'none';
        });
        details.style.display = 'block';
    }
}

// 搜索模型功能
function searchModels() {
    const searchInput = document.getElementById('model-search');
    const searchTerm = searchInput.value.toLowerCase().trim();
    const clearBtn = document.querySelector('.clear-search');
    const searchCount = document.getElementById('search-count');
    
    // 显示/隐藏清除按钮
    clearBtn.style.display = searchTerm ? 'block' : 'none';

    let visibleCount = 0;
    let totalCount = 0;

    // 获取所有模型行
    const allRows = document.querySelectorAll('.models-table tbody tr');
    
    allRows.forEach(row => {
        // 获取模型列表中的小模型名称
        let modelListNames = '';
        const modelListDetails = row.querySelector('.model-list-details');
        if (modelListDetails) {
            const modelItems = modelListDetails.querySelectorAll('li');
            modelItems.forEach(item => {
                modelListNames += item.textContent.toLowerCase() + ' ';
                totalCount++; // 增加小模型计数
            });
        } else {
            totalCount++; // 如果没有小模型列表，只增加1
        }
        
        const modelName = row.querySelector('.model-cell')?.textContent.toLowerCase() || '';
        const codebase = row.querySelector('.codebase-link')?.textContent.toLowerCase() || '';
        const modelType = row.querySelector('.type-badge')?.textContent.toLowerCase() || '';
        const runtime = row.querySelector('.runtime-badge')?.textContent.toLowerCase() || '';
                
        const matches = !searchTerm || 
            modelName.includes(searchTerm) || 
            codebase.includes(searchTerm) || 
            modelType.includes(searchTerm) || 
            runtime.includes(searchTerm) ||
            modelListNames.includes(searchTerm);

        if (matches) {
            row.style.display = '';
            // 如果有搜索词且匹配，计算匹配的小模型数量
            if (searchTerm && modelListDetails) {
                let rowVisibleCount = 0;
                const modelItems = modelListDetails.querySelectorAll('li');
                modelItems.forEach(item => {
                    if (item.textContent.toLowerCase().includes(searchTerm)) {
                        rowVisibleCount++;
                    }
                });
                // 如果模型本身信息匹配但小模型都不匹配，至少计数1
                if (rowVisibleCount === 0) {
                    visibleCount++;
                } else {
                    visibleCount += rowVisibleCount;
                }
            } else {
                visibleCount++;
            }
        } else {
            row.style.display = 'none';
        }
    });

    // 显示搜索结果计数
    if (searchTerm) {
        searchCount.style.display = 'block';
        // 统计匹配的模型名称数量（主模型数量）
        let matchedModelNameCount = 0;
        document.querySelectorAll('.models-table tbody tr').forEach(row => {
            if (row.style.display !== 'none') {
                matchedModelNameCount++;
            }
        });
        
        // 检查搜索词是否匹配了模型类型或运行时
        let isSearchingTypeOrRuntime = false;
        const allRows = document.querySelectorAll('.models-table tbody tr');
        allRows.forEach(row => {
            const modelType = row.querySelector('.type-badge')?.textContent.toLowerCase() || '';
            const runtime = row.querySelector('.runtime-badge')?.textContent.toLowerCase() || '';
            if (modelType.includes(searchTerm) || runtime.includes(searchTerm)) {
                isSearchingTypeOrRuntime = true;
            }
        });
        
        // 根据搜索内容类型显示不同的文案
        if (isSearchingTypeOrRuntime) {
            searchCount.textContent = i18next.t('searchResults', { count: visibleCount });
        } else {
            searchCount.textContent = i18next.t('searchResultsWithDetails', { modelCount: matchedModelNameCount, listCount: visibleCount });
        }
        
        // 自动展开所有包含结果的分类
        expandCategoriesWithResults();
    } else {
        searchCount.style.display = 'none';
        // 清除搜索时,折叠所有分类
        collapseAllCategories();
    }
}

// 清除搜索
function clearSearch() {
    const searchInput = document.getElementById('model-search');
    searchInput.value = '';
    searchModels();
}

// 展开包含搜索结果的分类
function expandCategoriesWithResults() {
    // 移除左侧导航栏的所有高亮
    document.querySelectorAll('.category-list > li > a').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelectorAll('.subcategory-list a').forEach(link => {
        link.classList.remove('active');
    });
    
    // 折叠左侧导航栏的所有子分类列表
    document.querySelectorAll('.subcategory-list').forEach(list => {
        list.classList.remove('expanded');
        const listButton = list.previousElementSibling;
        if (listButton && listButton.classList.contains('toggle-subcategories')) {
            const listIcon = listButton.querySelector('i');
            if (listIcon) {
                listIcon.className = 'fas fa-chevron-down';
            }
        }
    });
    
    const allTables = document.querySelectorAll('.models-table');
    
    allTables.forEach(table => {
        const visibleRows = Array.from(table.querySelectorAll('tbody tr'))
            .filter(row => row.style.display !== 'none');
        
        const subcategoryItem = table.closest('.subcategory-item');
        
        if (visibleRows.length > 0) {
            // 有结果：展开并显示子类别
            if (subcategoryItem) {
                subcategoryItem.style.display = '';
                const header = subcategoryItem.querySelector('.subcategory-header');
                const content = subcategoryItem.querySelector('.models-table-container');
                if (header && content) {
                    header.classList.add('active');
                    content.classList.add('expanded');
                }
            }
            
            // 展开大类
            const categoryCard = table.closest('.category-card');
            if (categoryCard) {
                categoryCard.style.display = '';
                const categoryHeader = categoryCard.querySelector('.category-header');
                const categoryContent = categoryCard.querySelector('.subcategory-panel');
                if (categoryHeader && categoryContent) {
                    categoryHeader.classList.add('active');
                    categoryContent.classList.add('expanded');
                }
            }
        } else {
            // 无结果：隐藏整个子类别
            if (subcategoryItem) {
                subcategoryItem.style.display = 'none';
            }
        }
    });
    
    // 检查每个大类是否有可见的子类别，如果没有则隐藏整个大类
    document.querySelectorAll('.category-card').forEach(card => {
        const visibleSubcategories = Array.from(card.querySelectorAll('.subcategory-item'))
            .filter(item => item.style.display !== 'none');
        
        if (visibleSubcategories.length === 0) {
            card.style.display = 'none';
        }
    });
    
    // 移除自动滚动功能，保持页面当前位置
    // 如需恢复自动滚动，请取消注释以下代码
    /*
    setTimeout(() => {
        const firstVisibleSubcategory = document.querySelector('.subcategory-item:not([style*="display: none"])');
        if (firstVisibleSubcategory && window.categoryPositions && window.categoryPositions.subcategories) {
            const subcategoryId = firstVisibleSubcategory.id;
            if (window.categoryPositions.subcategories[subcategoryId]) {
                window.scrollTo({
                    top: window.categoryPositions.subcategories[subcategoryId] - 120,
                    behavior: 'smooth'
                });
            }
        }
    }, 300);
    */
}

// 折叠所有分类
function collapseAllCategories() {
    // 移除左侧导航栏的所有高亮
    document.querySelectorAll('.category-list > li > a').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelectorAll('.subcategory-list a').forEach(link => {
        link.classList.remove('active');
    });
    
    // 折叠左侧导航栏的所有子分类列表
    document.querySelectorAll('.subcategory-list').forEach(list => {
        list.classList.remove('expanded');
        const listButton = list.previousElementSibling;
        if (listButton && listButton.classList.contains('toggle-subcategories')) {
            const listIcon = listButton.querySelector('i');
            if (listIcon) {
                listIcon.className = 'fas fa-chevron-down';
            }
        }
    });
    
    // 折叠所有子类别
    document.querySelectorAll('.subcategory-header').forEach(header => {
        header.classList.remove('active');
    });
    document.querySelectorAll('.models-table-container').forEach(content => {
        content.classList.remove('expanded');
    });
    
    // 折叠所有大类
    document.querySelectorAll('.category-header').forEach(header => {
        header.classList.remove('active');
    });
    document.querySelectorAll('.subcategory-panel').forEach(panel => {
        panel.classList.remove('expanded');
    });
    
    // 显示所有分类和子类别
    document.querySelectorAll('.category-card').forEach(card => {
        card.style.display = '';
    });
    document.querySelectorAll('.subcategory-item').forEach(item => {
        item.style.display = '';
    });
}

// 按运行环境过滤
let currentRuntimeFilter = 'all';
function filterByRuntime(runtime) {
    currentRuntimeFilter = runtime;
    
    // 更新按钮状态
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-filter="${runtime}"]`).classList.add('active');

    // 过滤模型行
    const allRows = document.querySelectorAll('.models-table tbody tr');
    let visibleCount = 0;
    
    allRows.forEach(row => {
        const runtimeBadge = row.querySelector('.runtime-badge');
        if (!runtimeBadge) return;
        
        const rowRuntime = runtimeBadge.textContent.toLowerCase();
        
        if (runtime === 'all' || rowRuntime.includes(runtime)) {
            row.style.display = '';
            visibleCount++;
        } else {
            row.style.display = 'none';
        }
    });

    // 如果有搜索词，重新应用搜索
    const searchTerm = document.getElementById('model-search').value;
    if (searchTerm) {
        searchModels();
    }
}

// 字母索引滚动
function scrollToLetter(letter) {
    const allRows = document.querySelectorAll('.models-table tbody tr');
    let found = false;

    for (let row of allRows) {
        const modelCell = row.querySelector('.model-cell');
        if (!modelCell) continue;
        
        const modelName = modelCell.textContent.trim().toUpperCase();
        let firstChar = modelName.charAt(0);
        
        // 处理数字
        if (letter === '0' && /[0-9]/.test(firstChar)) {
            found = true;
        } else if (firstChar === letter) {
            found = true;
        }
        
        if (found) {
            // 展开相关的类别
            const subcategoryItem = row.closest('.subcategory-item');
            if (subcategoryItem) {
                const header = subcategoryItem.querySelector('.subcategory-header');
                const content = subcategoryItem.querySelector('.models-table-container');
                if (header && content) {
                    header.classList.add('active');
                    content.classList.add('expanded');
                }
            }
            
            const categoryCard = row.closest('.category-card');
            if (categoryCard) {
                const categoryHeader = categoryCard.querySelector('.category-header');
                const categoryContent = categoryCard.querySelector('.subcategory-panel');
                if (categoryHeader && categoryContent) {
                    categoryHeader.classList.add('active');
                    categoryContent.classList.add('expanded');
                }
            }
            
            // 滚动到该行
            // 使用预先计算的位置信息进行滚动（如果有的话）
            let scrollTarget = row.offsetTop - 200;
            
            // 检查该行所属的子类别是否有预先计算的位置
            if (subcategoryItem && window.categoryPositions && window.categoryPositions.subcategories) {
                const subcategoryId = subcategoryItem.id;
                if (window.categoryPositions.subcategories[subcategoryId]) {
                    scrollTarget = window.categoryPositions.subcategories[subcategoryId] - 120;
                }
            }
            
            window.scrollTo({
                top: scrollTarget,
                behavior: 'smooth'
            });
            
            // 高亮该行
            row.classList.add('highlight-row');
            setTimeout(() => {
                row.classList.remove('highlight-row');
            }, 2000);
            
            break;
        }
    }
    
    if (!found) {
        alert(i18next.t('noModelsFound', { letter: letter }));
    }
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function () {
    // 设置侧边导航点击效果
    document.querySelectorAll('.category-list > li > a').forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);

            // 移除所有active类
            document.querySelectorAll('.category-list > li > a').forEach(a => {
                a.classList.remove('active');
            });

            // 添加当前active类
            this.classList.add('active');

            // 滚动到目标位置
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                // 获取目标大类对应的左侧导航栏子列表ID
                const categoryIdMap = {
                    'cv-models': 'cv-sub',
                    'nlp-models': 'nlp-sub',
                    'llm-models': 'llm-sub',
                    'vlm-models': 'vlm-sub',
                    'alm-models': 'alm-sub'
                };
                const targetNavListId = categoryIdMap[targetId];
                
                // 先折叠所有其他大类
                document.querySelectorAll('.subcategory-panel').forEach(panel => {
                    if (panel.id !== targetElement.querySelector('.subcategory-panel')?.id) {
                        panel.classList.remove('expanded');
                        const panelHeader = panel.previousElementSibling;
                        if (panelHeader) {
                            panelHeader.classList.remove('active');
                        }
                    }
                });
                
                // 折叠左侧导航栏中所有其他大类的子模型列表
                document.querySelectorAll('.subcategory-list').forEach(list => {
                    if (list.id !== targetNavListId) {
                        list.classList.remove('expanded');
                        const listButton = list.previousElementSibling;
                        if (listButton && listButton.classList.contains('toggle-subcategories')) {
                            const listIcon = listButton.querySelector('i');
                            if (listIcon) {
                                listIcon.className = 'fas fa-chevron-down';
                            }
                        }
                    }
                });
                
                // 展开左侧导航栏对应的子分类列表
                if (targetNavListId) {
                    const targetNavList = document.getElementById(targetNavListId);
                    if (targetNavList) {
                        targetNavList.classList.add('expanded');
                        const navButton = targetNavList.previousElementSibling;
                        if (navButton && navButton.classList.contains('toggle-subcategories')) {
                            const navIcon = navButton.querySelector('i');
                            if (navIcon) {
                                navIcon.className = 'fas fa-chevron-up';
                            }
                        }
                    }
                }
                
                // 展开目标大类
                const categoryCard = targetElement.closest('.category-card');
                if (categoryCard) {
                    const categoryHeader = categoryCard.querySelector('.category-header');
                    const categoryContent = categoryCard.querySelector('.subcategory-panel');

                    categoryHeader.classList.add('active');
                    categoryContent.classList.add('expanded');
                }

                // 使用预先计算的位置信息进行滚动
                const targetTop = window.categoryPositions && window.categoryPositions.categories[targetId] 
                    ? window.categoryPositions.categories[targetId] - 120 
                    : targetElement.offsetTop - 120;
                
                window.scrollTo({
                    top: targetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    // 设置子类别导航点击效果
    document.querySelectorAll('.subcategory-list a').forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);

            // 高亮当前点击的子类别链接
            document.querySelectorAll('.subcategory-list a').forEach(a => {
                a.classList.remove('active');
            });
            this.classList.add('active');

            // 滚动到目标位置
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                const targetCategoryCard = targetElement.closest('.category-card');
                
                // 获取当前点击的子类别所属的左侧导航栏列表ID
                const currentNavList = this.closest('.subcategory-list');
                const currentNavListId = currentNavList ? currentNavList.id : null;
                
                // 第一步：先折叠所有内容（包括其他大类和它们内部的所有子类别）
                document.querySelectorAll('.subcategory-panel').forEach(panel => {
                    const panelCard = panel.closest('.category-card');
                    if (panelCard !== targetCategoryCard) {
                        panel.classList.remove('expanded');
                        const panelHeader = panel.previousElementSibling;
                        if (panelHeader) {
                            panelHeader.classList.remove('active');
                        }
                        
                        // 关键：折叠该大类内的所有子类别
                        panel.querySelectorAll('.models-table-container').forEach(container => {
                            container.classList.remove('expanded');
                            const containerHeader = container.previousElementSibling;
                            if (containerHeader) {
                                containerHeader.classList.remove('active');
                            }
                        });
                    }
                });
                
                // 折叠同一大类下的所有其他子类别（不包括目标子类别）
                const parentPanel = targetElement.closest('.subcategory-panel');
                const targetContainer = targetElement.querySelector('.models-table-container');
                if (parentPanel) {
                    parentPanel.querySelectorAll('.models-table-container').forEach(container => {
                        if (container !== targetContainer) {
                            container.classList.remove('expanded');
                            const containerHeader = container.previousElementSibling;
                            if (containerHeader) {
                                containerHeader.classList.remove('active');
                            }
                            // 关闭该子类别内所有已展开的模型列表
                            container.querySelectorAll('.model-list-details').forEach(details => {
                                details.style.display = 'none';
                            });
                        }
                    });
                }
                
                // 同时关闭目标子类别内所有已展开的模型列表
                if (targetContainer) {
                    targetContainer.querySelectorAll('.model-list-details').forEach(details => {
                        details.style.display = 'none';
                    });
                }
                
                // 第二步：等待折叠动画完成后，展开目标大类
                setTimeout(() => {
                    if (targetCategoryCard) {
                        const categoryHeader = targetCategoryCard.querySelector('.category-header');
                        const categoryContent = targetCategoryCard.querySelector('.subcategory-panel');
                        if (categoryHeader && categoryContent) {
                            categoryHeader.classList.add('active');
                            categoryContent.classList.add('expanded');
                        }
                    }
                    
                    // 第三步：再等待一下，展开目标子类别
                    setTimeout(() => {
                        const subcategoryHeader = targetElement.querySelector('.subcategory-header');
                        const subcategoryContent = targetElement.querySelector('.models-table-container');

                        if (subcategoryHeader && subcategoryContent) {
                            subcategoryHeader.classList.add('active');
                            subcategoryContent.classList.add('expanded');
                        }
                        
                        // 第四步：最后滚动到目标位置
                        setTimeout(() => {
                            const scrollTarget = subcategoryHeader || targetElement;
                            
                            // 使用预先计算的位置信息进行滚动
                            const targetTop = window.categoryPositions && window.categoryPositions.subcategories[targetId]
                                ? window.categoryPositions.subcategories[targetId] - 120
                                : scrollTarget.getBoundingClientRect().top + window.pageYOffset - 120;
                            
                            window.scrollTo({
                                top: targetTop,
                                behavior: 'smooth'
                            });
                        }, 50);
                    }, 50);
                }, 450);
            }
        });
    });

    // 初始化时打开第一个大类的第一个子类别
    // 已禁用：默认全部折叠
    // const firstSubcategory = document.querySelector('.subcategory-item');
    // if (firstSubcategory) {
    //     const header = firstSubcategory.querySelector('.subcategory-header');
    //     const content = firstSubcategory.querySelector('.models-table-container');
    //     if (header && content) {
    //         header.classList.add('active');
    //         content.classList.add('expanded');
    //     }
    // }
});

// 返回顶部功能
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// 显示/隐藏返回顶部按钮
window.addEventListener('scroll', function() {
    const backToTopButton = document.getElementById('backToTop');
    if (backToTopButton) {
        if (window.pageYOffset > 300) {
            backToTopButton.classList.add('show');
        } else {
            backToTopButton.classList.remove('show');
        }
    }
});

// 计算所有类别和子类别相对页面顶部的位置
function calculateAllPositions() {
    const positions = {
        categories: {},
        subcategories: {}
    };

    // 大类ID列表
    const categoryIds = ['cv-models', 'nlp-models', 'llm-models', 'vlm-models', 'alm-models'];
    
    // 子类别ID列表
    const subcategoryIds = [
        // CV子类别
        'cv-classification', 'cv-detection', 'cv-detection3d', 'cv-segmentation',
        'cv-face-alignment', 'cv-face-detection', 'cv-face-quality', 'cv-face-recognize',
        'cv-facial-attribute', 'cv-image-colorization', 'cv-image-compress', 'cv-image-retrieval',
        'cv-low-light-image-enhancement', 'cv-mot', 'cv-pose', 'cv-reid',
        'cv-salient-object-detection', 'cv-super-resolution', 'cv-text-detection', 'cv-text-recognition',
        // NLP子类别
        'nlp-information-extraction', 'nlp-named-entity-recognition', 'nlp-question-answering',
        'nlp-sentence-classification', 'nlp-text2vec', 'nlp-reranker',
        // LLM子类别
        'llm-general', 'llm-domain',
        // VLM子类别
        'vlm-general', 'vlm-domain',
        // ALM子类别
        'alm-general', 'alm-domain'
    ];

    // 计算大类位置
    categoryIds.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            positions.categories[id] = element.offsetTop;
        }
    });

    // 计算子类别位置
    subcategoryIds.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            positions.subcategories[id] = element.offsetTop;
        }
    });

    return positions;
}

// 打印所有位置信息
function printAllPositions() {
    const positions = calculateAllPositions();
    
    console.log('=== 大类位置 (相对页面顶部) ===');
    for (const [id, offset] of Object.entries(positions.categories)) {
        console.log(`${id}: ${offset}px`);
    }
    
    console.log('\n=== 子类别位置 (相对页面顶部) ===');
    for (const [id, offset] of Object.entries(positions.subcategories)) {
        console.log(`${id}: ${offset}px`);
    }
    
    return positions;
}

// 在页面加载完成后自动计算并打印位置
window.addEventListener('load', function() {
    setTimeout(() => {
        try {
            const positions = printAllPositions();
            
            // 将位置信息保存到全局变量，方便其他函数使用
            window.categoryPositions = positions;
        } catch (error) {
            console.error('计算位置信息时出错:', error);
            // 如果出错，设置为空对象，避免后续代码出错
            window.categoryPositions = { categories: {}, subcategories: {} };
        }
    }, 1000);
});