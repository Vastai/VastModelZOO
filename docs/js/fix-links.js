/**
 * 修复GitHub Pages上的Markdown链接
 * 将本地相对路径转换为GitHub仓库的绝对路径
 */

(function() {
    'use strict';
    
    // ============ 配置区域 ============
    // 从当前URL自动检测GitHub信息
    const pathname = window.location.pathname;
    const pathParts = pathname.split('/').filter(p => p);
    
    // GitHub配置
    const GITHUB_CONFIG = {
        user: 'jane-vastai',           // 从URL检测: jane-vastai.github.io
        repo: pathParts[0] || 'VastModelZOO',  // 仓库名
        branch: 'main'                  // 默认分支
    };
    
    // 构建GitHub base URL
    const GITHUB_BASE_URL = `https://github.com/${GITHUB_CONFIG.user}/${GITHUB_CONFIG.repo}/blob/${GITHUB_CONFIG.branch}`;
    
    console.log('🔧 Markdown Link Fixer initialized');
    console.log('📦 Repository:', `${GITHUB_CONFIG.user}/${GITHUB_CONFIG.repo}`);
    
    /**
     * 修复单个链接
     */
    function fixLink(link) {
        const href = link.getAttribute('href');
        
        // 跳过条件
        if (!href || 
            href.startsWith('http://') || 
            href.startsWith('https://') ||
            href.startsWith('#') ||
            href.startsWith('javascript:') ||
            !href.endsWith('.md')) {
            return false;
        }
        
        // 处理相对路径
        let cleanPath = href;
        
        // 移除开头的 ../
        while (cleanPath.startsWith('../')) {
            cleanPath = cleanPath.substring(3);
        }
        
        // 移除开头的 ./
        if (cleanPath.startsWith('./')) {
            cleanPath = cleanPath.substring(2);
        }
        
        // 构建完整的GitHub URL
        const githubURL = `${GITHUB_BASE_URL}/${cleanPath}`;
        
        // 更新链接属性
        link.setAttribute('href', githubURL);
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
        
        // 添加提示标记
        if (!link.querySelector('.external-link-icon')) {
            const icon = document.createElement('i');
            icon.className = 'fas fa-external-link-alt external-link-icon';
            icon.style.cssText = 'margin-left: 5px; font-size: 0.8em; opacity: 0.6;';
            link.appendChild(icon);
        }
        
        return true;
    }
    
    /**
     * 修复所有Markdown链接
     */
    function fixAllLinks() {
        // 查找所有.md链接
        const links = document.querySelectorAll('a[href$=".md"]');
        let fixedCount = 0;
        
        links.forEach(link => {
            if (fixLink(link)) {
                fixedCount++;
            }
        });
        
        if (fixedCount > 0) {
            console.log(`✅ Fixed ${fixedCount} Markdown link(s)`);
        }
        
        return fixedCount;
    }
    
    /**
     * 监听动态内容变化
     */
    function observeChanges() {
        const observer = new MutationObserver((mutations) => {
            let hasNewLinks = false;
            mutations.forEach(mutation => {
                if (mutation.addedNodes.length > 0) {
                    hasNewLinks = true;
                }
            });
            
            if (hasNewLinks) {
                fixAllLinks();
            }
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        console.log('👀 Watching for dynamic content changes');
    }
    
    /**
     * 初始化
     */
    function init() {
        // 立即执行一次
        fixAllLinks();
        
        // 监听动态内容
        observeChanges();
        
        // 为搜索功能添加支持
        if (window.searchModels) {
            const originalSearch = window.searchModels;
            window.searchModels = function() {
                originalSearch.apply(this, arguments);
                setTimeout(fixAllLinks, 100);
            };
        }
    }
    
    // 页面加载完成后执行
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    // 暴露全局函数供调试使用
    window.fixMarkdownLinks = fixAllLinks;
    
})();
