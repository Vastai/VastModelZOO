// 页面加载完成后隐藏加载动画
window.addEventListener('load', function() {
    const pageLoader = document.getElementById('pageLoader');
    if (pageLoader) {
        setTimeout(() => {
            pageLoader.classList.add('hidden');
            document.body.classList.remove('loading');
        }, 300);
    }
});

// 切换大类内容显示/隐藏
function toggleCategory(categoryId) {
    const content = document.getElementById(categoryId);
    const header = content.previousElementSibling;
    
    const isExpanding = !content.classList.contains('expanded');
    
    if (isExpanding) {
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
                });
            }
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
        const navIdMap = {
            'cv-content': 'cv-sub',
            'nlp-content': 'nlp-sub',
            'llm-content': 'llm-sub',
            'vlm-content': 'vlm-sub'
        };
        const categoryLinkMap = {
            'cv-content': '#cv-models',
            'nlp-content': '#nlp-models',
            'llm-content': '#llm-models',
            'vlm-content': '#vlm-models'
        };
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
        // 折叠当前大类时,同时折叠其内部的所有子类别
        content.querySelectorAll('.models-table-container').forEach(container => {
            container.classList.remove('expanded');
            const containerHeader = container.previousElementSibling;
            if (containerHeader) {
                containerHeader.classList.remove('active');
            }
        });
        
        // 移除左侧导航栏对应类别的高亮
        const targetLink = categoryLinkMap[categoryId];
        if (targetLink) {
            const targetLinkElement = document.querySelector(`.category-list > li > a[href="${targetLink}"]`);
            if (targetLinkElement) {
                targetLinkElement.classList.remove('active');
            }
        }
    }

    content.classList.toggle('expanded');
    header.classList.toggle('active');
    
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
                'vlm-models': 'vlm-sub'
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
        'vlm-sub': 'vlm-content'
    };
    const subToCategoryLinkMap = {
        'cv-sub': '#cv-models',
        'nlp-sub': '#nlp-models',
        'llm-sub': '#llm-models',
        'vlm-sub': '#vlm-models'
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
        totalCount++;
        const modelName = row.querySelector('.model-cell')?.textContent.toLowerCase() || '';
        const codebase = row.querySelector('.codebase-link')?.textContent.toLowerCase() || '';
        const modelType = row.querySelector('.type-badge')?.textContent.toLowerCase() || '';
        const runtime = row.querySelector('.runtime-badge')?.textContent.toLowerCase() || '';
        
        const matches = !searchTerm || 
            modelName.includes(searchTerm) || 
            codebase.includes(searchTerm) || 
            modelType.includes(searchTerm) || 
            runtime.includes(searchTerm);

        if (matches) {
            row.style.display = '';
            visibleCount++;
        } else {
            row.style.display = 'none';
        }
    });

    // 显示搜索结果计数
    if (searchTerm) {
        searchCount.style.display = 'block';
        searchCount.textContent = `找到 ${visibleCount} 个结果 (共 ${totalCount} 个模型)`;
        
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
    
    // 滚动到第一个有结果的子类别
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
        alert(`没有找到以 "${letter}" 开头的模型`);
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
                    'vlm-models': 'vlm-sub'
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
    const categoryIds = ['cv-models', 'nlp-models', 'llm-models', 'vlm-models'];
    
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
        'vlm-general', 'vlm-domain'
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