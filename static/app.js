/**
 * CAG-OKF Knowledge Studio 2.0 — Admin Dashboard Application Logic
 * 100% Self-Contained. Zero external dependencies.
 */

let allDocuments = [];
let currentDoc = null;
let currentStatusFilter = 'ALL';
let editorMode = 'visual'; // 'visual' or 'raw'

// ==========================================================================
// Embedded Lightweight Markdown Parser (Zero external CDN dependency)
// ==========================================================================

function parseMarkdown(md) {
  if (!md) return '<p style="color:var(--text-dim);font-style:italic;">Nenhum conteúdo digitado ainda.</p>';
  
  // Escape raw HTML characters to prevent XSS
  let html = md
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Fenced Code Blocks (```lang ... ```)
  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
  });

  // Inline code (`code`)
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headings
  html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Blockquotes (> quote)
  html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

  // Horizontal Rule (---)
  html = html.replace(/^---$/gim, '<hr style="border:0;border-top:1px solid var(--border-card);margin:16px 0;" />');

  // Bold & Italic
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  html = html.replace(/_([^_]+)_/g, '<em>$1</em>');

  // Lists (- item or * item)
  html = html.replace(/^\s*[-*]\s+(.*$)/gim, '<ul><li>$1</li></ul>');
  html = html.replace(/<\/ul>\s*<ul>/g, ''); // Join consecutive lists

  // Ordered lists (1. item)
  html = html.replace(/^\s*\d+\.\s+(.*$)/gim, '<ol><li>$1</li></ol>');
  html = html.replace(/<\/ol>\s*<ol>/g, ''); // Join consecutive ordered lists

  // Safe Links [text](url) — Blocks javascript: and data: URI vectors
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, text, url) => {
    const cleanUrl = url.trim();
    if (/^(https?:\/\/|mailto:|#|\/)/i.test(cleanUrl)) {
      return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" style="color:#818cf8;text-decoration:underline;">${text}</a>`;
    }
    return `<span>${text}</span>`;
  });

  // Paragraphs (lines separated by double newlines)
  const blocks = html.split(/\n\s*\n/);
  const parsedBlocks = blocks.map(b => {
    b = b.trim();
    if (!b) return '';
    if (b.startsWith('<h') || b.startsWith('<pre') || b.startsWith('<blockquote') || 
        b.startsWith('<ul') || b.startsWith('<ol') || b.startsWith('<hr')) {
      return b;
    }
    return `<p>${b.replace(/\n/g, '<br/>')}</p>`;
  });

  return parsedBlocks.filter(Boolean).join('\n');
}

function renderMarkdownPreview() {
  const bodyInput = document.getElementById('doc-body-input');
  const preview = document.getElementById('markdown-preview');
  if (!bodyInput || !preview) return;

  const content = bodyInput.value;
  try {
    if (typeof window.marked !== 'undefined' && typeof window.marked.parse === 'function') {
      preview.innerHTML = window.marked.parse(content || '*Nenhum conteúdo digitado ainda.*');
    } else {
      preview.innerHTML = parseMarkdown(content);
    }
  } catch (e) {
    preview.innerHTML = parseMarkdown(content);
  }
}

// ==========================================================================
// Lifecycle Initialization
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  fetchDocuments();
  fetchStats();
  // Poll stats every 5 seconds to keep telemetry fresh
  setInterval(fetchStats, 5000);

  // Hook live markdown typing
  const bodyInput = document.getElementById('doc-body-input');
  if (bodyInput) {
    bodyInput.addEventListener('input', () => {
      renderMarkdownPreview();
      const approxTokens = Math.max(1, Math.round(bodyInput.value.length / 4));
      const badge = document.getElementById('doc-token-badge');
      if (badge) badge.innerText = `~${approxTokens} tokens`;
    });
  }
});

// ==========================================================================
// Admin Sidebar Tab Switching
// ==========================================================================

const TAB_TITLES = {
  editor: 'Documentos & Regras Arquiteturais',
  playground: 'CAG Playground & Inferência em Memória',
  stats: 'Telemetria & Cache em RAM',
  integration: 'Endpoints de Integração & Model Context Protocol'
};

function switchTab(tab) {
  // Update sidebar navigation buttons
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`nav-${tab}-btn`);
  if (activeBtn) activeBtn.classList.add('active');

  // Update breadcrumb title
  const breadcrumb = document.getElementById('breadcrumb-title');
  if (breadcrumb && TAB_TITLES[tab]) {
    breadcrumb.innerText = TAB_TITLES[tab];
  }

  // Update tab pane viewports
  const panes = ['editor', 'playground', 'stats', 'integration'];
  panes.forEach(p => {
    const pane = document.getElementById(`tab-${p}`);
    if (pane) {
      if (p === tab) {
        pane.classList.remove('hidden');
        pane.style.display = (p === 'editor') ? 'flex' : 'flex';
      } else {
        pane.classList.add('hidden');
        pane.style.display = 'none';
      }
    }
  });

  if (tab === 'stats') {
    fetchStats();
  }
}

// ==========================================================================
// Document Management & REST Calls
// ==========================================================================

async function fetchDocuments() {
  try {
    const res = await fetch('/api/documents');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allDocuments = await res.json();
    
    // Update sidebar doc badge
    const badge = document.getElementById('sidebar-docs-badge');
    if (badge) badge.innerText = allDocuments.length;

    renderDocumentList();

    if (allDocuments.length > 0) {
      if (!currentDoc) {
        selectDocument(allDocuments[0].id);
      } else {
        const updated = allDocuments.find(d => d.id === currentDoc.id);
        if (updated) selectDocument(updated.id);
      }
    }
  } catch (err) {
    showToast('Erro ao carregar documentos da memória: ' + err.message, '⚠️');
  }
}

function renderDocumentList() {
  const container = document.getElementById('document-list');
  if (!container) return;

  const searchInput = document.getElementById('doc-search');
  const search = searchInput ? searchInput.value.toLowerCase().trim() : '';

  const filtered = allDocuments.filter(d => {
    const status = (d.status || 'APPROVED').toUpperCase();
    const matchStatus = currentStatusFilter === 'ALL' || status === currentStatusFilter;
    const matchSearch = !search || 
      (d.id && d.id.toLowerCase().includes(search)) || 
      (d.title && d.title.toLowerCase().includes(search)) ||
      (d.tags && d.tags.some(t => t.toLowerCase().includes(search))) ||
      (d.target_services && d.target_services.some(s => s.toLowerCase().includes(search)));
    return matchStatus && matchSearch;
  });

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="padding: 30px 15px; text-align: center; color: var(--text-dim); font-size: 11px;">
        Nenhum documento encontrado.
      </div>
    `;
    return;
  }

  container.innerHTML = filtered.map(d => {
    const isSelected = currentDoc && currentDoc.id === d.id;
    const activeClass = isSelected ? 'active' : '';
    
    const status = (d.status || 'APPROVED').toUpperCase();
    let badgeClass = 'approved';
    if (status === 'DRAFT') badgeClass = 'draft';
    if (status === 'DEPRECATED') badgeClass = 'deprecated';

    const services = (d.target_services || []).slice(0, 2).map(s => 
      `<span class="service-tag-pill">${s}</span>`
    ).join(' ');

    return `
      <div onclick="selectDocument('${d.id}')" class="doc-card ${activeClass}">
        <div class="doc-card-header">
          <span class="doc-card-id">${d.id}</span>
          <span class="status-badge ${badgeClass}">${status}</span>
        </div>
        <div class="doc-card-title">${d.title || 'Sem título cadastrado'}</div>
        <div class="doc-card-footer">
          <div style="display: flex; align-items: center; gap: 4px; overflow: hidden;">
            <span class="service-tag-pill" style="color: #818cf8; background-color: rgba(99, 102, 241, 0.12); border-color: rgba(99, 102, 241, 0.25);">${d.type || 'rule'}</span>
            ${services}
          </div>
          <span style="font-family: var(--font-mono); font-weight: 600; color: #64748b; flex-shrink: 0;">${d.token_count || 0} tok</span>
        </div>
      </div>
    `;
  }).join('');
}

function setStatusFilter(status) {
  currentStatusFilter = status;
  document.querySelectorAll('.filter-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.status === status);
  });
  renderDocumentList();
}

function filterDocumentList() {
  renderDocumentList();
}

async function selectDocument(docId) {
  try {
    const res = await fetch(`/api/documents/${docId}`);
    if (!res.ok) throw new Error('Documento não encontrado');
    const data = await res.json();
    currentDoc = { id: docId, ...data };

    populateEditor(currentDoc);
    renderDocumentList();
  } catch (err) {
    showToast('Falha ao abrir documento: ' + err.message, '⚠️');
  }
}

function populateEditor(doc) {
  if (!doc) return;
  const meta = doc.metadata || {};

  const badge = document.getElementById('current-doc-badge');
  if (badge) badge.innerText = meta.id || doc.id || 'NO-ID';

  const titleInput = document.getElementById('editor-title');
  if (titleInput) titleInput.value = meta.title || doc.title || '';

  // Visual Form inputs
  const idInput = document.getElementById('doc-id-input');
  if (idInput) idInput.value = meta.id || doc.id || '';

  const typeInput = document.getElementById('doc-type-input');
  if (typeInput) typeInput.value = meta.type || doc.type || 'rule';

  const statusInput = document.getElementById('doc-status-input');
  if (statusInput) statusInput.value = (meta.status || doc.status || 'APPROVED').toUpperCase();

  const validUntilInput = document.getElementById('doc-valid-until-input');
  if (validUntilInput) validUntilInput.value = meta.valid_until || doc.valid_until || '';

  const servicesInput = document.getElementById('doc-services-input');
  if (servicesInput) servicesInput.value = (meta.target_services || doc.target_services || []).join(', ');

  const languagesInput = document.getElementById('doc-languages-input');
  if (languagesInput) languagesInput.value = (meta.target_languages || doc.target_languages || []).join(', ');

  const tagsInput = document.getElementById('doc-tags-input');
  if (tagsInput) tagsInput.value = (meta.tags || doc.tags || []).join(', ');

  // Body & Raw
  const bodyInput = document.getElementById('doc-body-input');
  if (bodyInput) bodyInput.value = doc.body || '';

  const rawInput = document.getElementById('raw-content-input');
  if (rawInput) rawInput.value = doc.raw_content || '';

  const tokenBadge = document.getElementById('doc-token-badge');
  if (tokenBadge) tokenBadge.innerText = `${doc.token_count || 0} tokens pré-calculados`;

  renderMarkdownPreview();
}

function setEditorMode(mode) {
  editorMode = mode;
  
  const visualBtn = document.getElementById('mode-visual-btn');
  const rawBtn = document.getElementById('mode-raw-btn');
  if (visualBtn) visualBtn.classList.toggle('active', mode === 'visual');
  if (rawBtn) rawBtn.classList.toggle('active', mode === 'raw');

  const visualContainer = document.getElementById('visual-editor-container');
  const rawContainer = document.getElementById('raw-editor-container');

  if (mode === 'visual') {
    if (visualContainer) { visualContainer.classList.remove('hidden'); visualContainer.style.display = 'flex'; }
    if (rawContainer) { rawContainer.classList.add('hidden'); rawContainer.style.display = 'none'; }
  } else {
    syncFormToRaw();
    if (visualContainer) { visualContainer.classList.add('hidden'); visualContainer.style.display = 'none'; }
    if (rawContainer) { rawContainer.classList.remove('hidden'); rawContainer.style.display = 'flex'; }
  }
}

function syncFormToRaw() {
  const id = document.getElementById('doc-id-input').value || 'WIKI-NEW';
  const title = document.getElementById('editor-title').value;
  const type = document.getElementById('doc-type-input').value;
  const status = document.getElementById('doc-status-input').value;
  const validUntil = document.getElementById('doc-valid-until-input').value;
  const services = document.getElementById('doc-services-input').value.split(',').map(s => s.trim()).filter(Boolean);
  const languages = document.getElementById('doc-languages-input').value.split(',').map(s => s.trim()).filter(Boolean);
  const tags = document.getElementById('doc-tags-input').value.split(',').map(s => s.trim()).filter(Boolean);
  const body = document.getElementById('doc-body-input').value;

  const raw = [
    '---',
    `id: "${id}"`,
    `type: "${type}"`,
    `title: "${title}"`,
    `status: "${status}"`,
    services.length ? `target_services: ${JSON.stringify(services)}` : '',
    languages.length ? `target_languages: ${JSON.stringify(languages)}` : '',
    tags.length ? `tags: ${JSON.stringify(tags)}` : '',
    validUntil ? `valid_until: "${validUntil}"` : '',
    '---',
    '',
    body
  ].filter(l => l !== '').join('\n');

  const rawInput = document.getElementById('raw-content-input');
  if (rawInput) rawInput.value = raw;
}

function createNewDocument() {
  const newId = `WIKI-RULE-00${allDocuments.length + 1}`;
  const newDoc = {
    id: newId,
    metadata: {
      id: newId,
      type: 'rule',
      title: 'Nova Regra Corporativa',
      status: 'APPROVED',
      target_services: ['pagamentos-service'],
      target_languages: ['golang'],
      tags: ['security', 'database']
    },
    body: '# Especificação da Regra\n\nDescreva a política ou padrão técnico aqui...',
    token_count: 0
  };
  currentDoc = newDoc;
  populateEditor(newDoc);
  switchTab('editor');
  showToast('Novo documento pronto para edição.', '✨');
}

async function saveCurrentDocument() {
  const persistChk = document.getElementById('persist-disk-chk');
  const persistToDisk = persistChk ? persistChk.checked : true;

  try {
    let res;
    if (editorMode === 'raw') {
      const rawContent = document.getElementById('raw-content-input').value;
      res = await fetch('/api/documents/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw_content: rawContent, persist_to_disk: persistToDisk })
      });
    } else {
      const id = document.getElementById('doc-id-input').value.trim();
      if (!id) {
        showToast('Identificador Único (ID) é obrigatório.', '⚠️');
        return;
      }
      const title = document.getElementById('editor-title').value.trim();
      const type = document.getElementById('doc-type-input').value;
      const status = document.getElementById('doc-status-input').value;
      const validUntil = document.getElementById('doc-valid-until-input').value || null;
      const services = document.getElementById('doc-services-input').value.split(',').map(s => s.trim()).filter(Boolean);
      const languages = document.getElementById('doc-languages-input').value.split(',').map(s => s.trim()).filter(Boolean);
      const tags = document.getElementById('doc-tags-input').value.split(',').map(s => s.trim()).filter(Boolean);
      const body = document.getElementById('doc-body-input').value;

      const payload = {
        id,
        title,
        type,
        status,
        valid_until: validUntil,
        target_services: services,
        target_languages: languages,
        tags,
        body,
        persist_to_disk: persistToDisk
      };

      res = await fetch(`/api/documents/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    }

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    const result = await res.json();

    showToast(`Salvo na memória RAM! (${result.token_count || 0} tokens)`, '💾');
    await fetchDocuments();
    await fetchStats();
  } catch (err) {
    showToast('Falha ao salvar: ' + err.message, '⚠️');
  }
}

async function deleteCurrentDocument() {
  if (!currentDoc) {
    showToast('Nenhum documento selecionado para exclusão.', '⚠️');
    return;
  }
  if (!confirm(`Deseja realmente excluir '${currentDoc.id}' da memória e disco?`)) return;

  try {
    const res = await fetch(`/api/documents/${currentDoc.id}?delete_from_disk=true`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Erro ao excluir documento');
    showToast(`Documento ${currentDoc.id} excluído com sucesso.`, '🗑️');
    currentDoc = null;
    await fetchDocuments();
    await fetchStats();
  } catch (err) {
    showToast('Falha ao excluir: ' + err.message, '⚠️');
  }
}

// ==========================================================================
// Key-Value Cache Operations
// ==========================================================================

async function rebuildKVCache() {
  const btn = document.getElementById('rebuild-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span>⏳ Recompilando...</span>`;
  }

  try {
    const res = await fetch('/api/cache/rebuild', { method: 'POST' });
    if (!res.ok) throw new Error('Falha na recompilação do cache');
    const data = await res.json();

    showToast(`KV Cache recompilado em ${data.duration_ms}ms! (${data.total_tokens} tokens)`, '⚡');
    await fetchStats();
  } catch (err) {
    showToast('Erro ao recompilar cache: ' + err.message, '⚠️');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<span>⚡ Recompilar Cache</span>`;
    }
  }
}

async function fetchStats() {
  try {
    const res = await fetch('/api/cache/stats');
    if (!res.ok) return;
    const stats = await res.json();

    // Update Sidebar Footer counters
    const sidebarDocs = document.getElementById('sidebar-docs');
    const sidebarTokens = document.getElementById('sidebar-tokens');
    const sidebarBadge = document.getElementById('sidebar-docs-badge');
    const hitRatioEl = document.getElementById('header-hit-ratio');

    if (sidebarDocs) sidebarDocs.innerText = stats.total_documents;
    if (sidebarTokens) sidebarTokens.innerText = stats.kv_cache_tokens || stats.total_pretokenized_tokens;
    if (sidebarBadge) sidebarBadge.innerText = stats.total_documents;
    if (hitRatioEl) hitRatioEl.innerText = `${stats.hit_ratio_percent || 100}%`;

    const dotWrapper = document.getElementById('cache-dot-wrapper');
    const statusText = document.getElementById('sidebar-cache-status');

    if (dotWrapper && statusText) {
      if (stats.is_dirty) {
        dotWrapper.classList.add('dirty');
        statusText.innerText = 'KV Cache Desatualizado';
      } else {
        dotWrapper.classList.remove('dirty');
        statusText.innerText = 'KV Cache Sincronizado';
      }
    }

    renderStatsTab(stats);
  } catch (err) {
    console.error('Stats poll error:', err);
  }
}

function renderStatsTab(stats) {
  const grid = document.getElementById('stats-grid');
  if (!grid) return;

  grid.innerHTML = `
    <div class="telemetry-card" style="border-top-color: #6366f1;">
      <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted);">
        <span>Total de Documentos</span>
        <span style="color:#818cf8; font-weight:700;">RAM</span>
      </div>
      <div style="font-family:var(--font-mono); font-size:24px; font-weight:800; color:#ffffff; margin-top:4px;">
        ${stats.total_documents}
      </div>
      <div style="font-size:11px; color:var(--text-dim); margin-top:2px;">regras e playbooks ativos</div>
    </div>

    <div class="telemetry-card" style="border-top-color: #06b6d4;">
      <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted);">
        <span>Tokens Pré-Compilados</span>
        <span style="color:#06b6d4; font-weight:700;">DynamicCache</span>
      </div>
      <div style="font-family:var(--font-mono); font-size:24px; font-weight:800; color:#38bdf8; margin-top:4px;">
        ${stats.kv_cache_tokens}
      </div>
      <div style="font-size:11px; color:var(--text-dim); margin-top:2px;">tensores Key-Value em RAM</div>
    </div>

    <div class="telemetry-card" style="border-top-color: #10b981;">
      <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted);">
        <span>Hit Ratio de Leitura</span>
        <span style="color:#10b981; font-weight:700;">O(1)</span>
      </div>
      <div style="font-family:var(--font-mono); font-size:24px; font-weight:800; color:#34d399; margin-top:4px;">
        ${stats.hit_ratio_percent}%
      </div>
      <div style="font-size:11px; color:var(--text-dim); margin-top:2px;">${stats.read_operations} leituras em sub-µs</div>
    </div>

    <div class="telemetry-card" style="border-top-color: #a855f7;">
      <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted);">
        <span>Tempo de Forward Pass</span>
        <span style="color:#a855f7; font-weight:700;">Prefill</span>
      </div>
      <div style="font-family:var(--font-mono); font-size:24px; font-weight:800; color:#c084fc; margin-top:4px;">
        ${stats.last_rebuild_duration_ms} ms
      </div>
      <div style="font-size:11px; color:var(--text-dim); margin-top:2px;">compilação dos tensores</div>
    </div>

    <div class="telemetry-card" style="border-top-color: #f59e0b;">
      <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted);">
        <span>Ocupação de RAM</span>
        <span style="color:#f59e0b; font-weight:700;">Memória</span>
      </div>
      <div style="font-family:var(--font-mono); font-size:24px; font-weight:800; color:#fbbf24; margin-top:4px;">
        ${stats.estimated_ram_kb} KB
      </div>
      <div style="font-size:11px; color:var(--text-dim); margin-top:2px;">tabelas de atenção e índices</div>
    </div>

    <div class="telemetry-card" style="border-top-color: #f43f5e;">
      <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted);">
        <span>Operações de Gravação</span>
        <span style="color:#f43f5e; font-weight:700;">Mutações</span>
      </div>
      <div style="font-family:var(--font-mono); font-size:24px; font-weight:800; color:#fb7185; margin-top:4px;">
        ${stats.write_operations}
      </div>
      <div style="font-size:11px; color:var(--text-dim); margin-top:2px;">atualizações e invalidações</div>
    </div>
  `;

  const servicesContainer = document.getElementById('indexed-services-container');
  if (servicesContainer) {
    const services = stats.services_indexed || [];
    const tags = stats.tags_indexed || [];
    
    servicesContainer.innerHTML = `
      <div style="margin-bottom: 14px;">
        <div style="font-size:11px; font-weight:700; text-transform:uppercase; color:var(--text-dim); margin-bottom:6px;">
          Serviços Indexados (${services.length}):
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:6px;">
          ${services.map(s => `<span class="service-tag-pill" style="padding:4px 10px; font-size:11px; color:#818cf8; border-color:rgba(99,102,241,0.3); background:rgba(99,102,241,0.1);">${s}</span>`).join('')}
        </div>
      </div>
      <div>
        <div style="font-size:11px; font-weight:700; text-transform:uppercase; color:var(--text-dim); margin-bottom:6px;">
          Tags Indexadas (${tags.length}):
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:6px;">
          ${tags.map(t => `<span class="service-tag-pill" style="padding:4px 10px; font-size:11px; color:#38bdf8; border-color:rgba(6,182,212,0.3); background:rgba(6,182,212,0.1);">${t}</span>`).join('')}
        </div>
      </div>
    `;
  }
}

// ==========================================================================
// CAG Playground Execution
// ==========================================================================

function setPlaygroundQuery(q) {
  const queryBox = document.getElementById('playground-query');
  if (queryBox) {
    queryBox.value = q;
    queryBox.focus();
  }
}

async function runPlaygroundQuery() {
  const queryBox = document.getElementById('playground-query');
  const query = queryBox ? queryBox.value.trim() : '';
  const maxTokens = parseInt(document.getElementById('playground-max-tokens').value) || 40;
  
  if (!query) {
    showToast('Digite uma consulta para executar.', '⚠️');
    return;
  }

  const btn = document.getElementById('query-run-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⏳ Executando...</span>';
  }

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, max_tokens: maxTokens, use_cache: true })
    });

    if (!res.ok) throw new Error('Erro na geração da resposta');
    const data = await res.json();

    const outputContainer = document.getElementById('playground-output-container');
    const benchBox = document.getElementById('benchmark-comparison-box');
    if (outputContainer) {
      outputContainer.classList.remove('hidden');
      outputContainer.style.display = 'flex';
    }
    if (benchBox) {
      benchBox.classList.add('hidden');
      benchBox.style.display = 'none';
    }

    document.getElementById('kpi-prefill-tokens').innerText = data.prefill_tokens_evaluated;
    document.getElementById('kpi-total-context').innerText = `de ${data.total_context_tokens} tokens totais`;
    document.getElementById('kpi-savings').innerText = `${data.token_savings_percent.toFixed(1)}%`;
    document.getElementById('kpi-latency').innerText = `${data.latency_seconds.toFixed(2)}s`;
    document.getElementById('kpi-throughput').innerText = data.tokens_per_second.toFixed(1);
    document.getElementById('playground-answer').innerText = data.answer;
  } catch (err) {
    showToast('Falha na inferência: ' + err.message, '⚠️');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>⚡ Executar com CAG</span>';
    }
  }
}

async function runBenchmarkQuery() {
  const queryBox = document.getElementById('playground-query');
  const query = queryBox ? queryBox.value.trim() : '';
  const maxTokens = parseInt(document.getElementById('playground-max-tokens').value) || 40;
  
  if (!query) {
    showToast('Digite uma consulta para o benchmark.', '⚠️');
    return;
  }

  const btn = document.getElementById('bench-run-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⏳ Comparando...</span>';
  }

  try {
    const res = await fetch('/api/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, max_tokens: maxTokens })
    });

    if (!res.ok) throw new Error('Erro no benchmark');
    const data = await res.json();

    const outputContainer = document.getElementById('playground-output-container');
    const benchBox = document.getElementById('benchmark-comparison-box');
    if (outputContainer) {
      outputContainer.classList.remove('hidden');
      outputContainer.style.display = 'flex';
    }
    if (benchBox) {
      benchBox.classList.remove('hidden');
      benchBox.style.display = 'block';
    }

    document.getElementById('bench-base-tokens').innerText = `${data.baseline.prefill_tokens} tokens`;
    document.getElementById('bench-base-time').innerText = `${data.baseline.duration_s.toFixed(2)}s`;
    document.getElementById('bench-cag-tokens').innerText = `${data.okf_cag.prefill_tokens} tokens`;
    document.getElementById('bench-cag-time').innerText = `${data.okf_cag.duration_s.toFixed(2)}s`;
    document.getElementById('bench-cag-savings').innerText = `${data.okf_cag.savings_percent.toFixed(1)}%`;

    document.getElementById('kpi-prefill-tokens').innerText = data.okf_cag.prefill_tokens;
    document.getElementById('kpi-total-context').innerText = `de ${data.baseline.prefill_tokens} tokens totais`;
    document.getElementById('kpi-savings').innerText = `${data.okf_cag.savings_percent.toFixed(1)}%`;
    document.getElementById('kpi-latency').innerText = `${data.okf_cag.duration_s.toFixed(2)}s`;
    document.getElementById('playground-answer').innerText = data.okf_cag.answer;
  } catch (err) {
    showToast('Falha no benchmark: ' + err.message, '⚠️');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>⚖️ Benchmark Comparativo</span>';
    }
  }
}

// ==========================================================================
// Toast Notification Helper
// ==========================================================================

let toastTimeout = null;

function showToast(message, icon = 'ℹ️') {
  const toast = document.getElementById('toast');
  if (!toast) return;

  const iconEl = document.getElementById('toast-icon');
  const msgEl = document.getElementById('toast-message');
  if (iconEl) iconEl.innerText = icon;
  if (msgEl) msgEl.innerText = message;

  toast.classList.add('show');

  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove('show');
  }, 3500);
}
