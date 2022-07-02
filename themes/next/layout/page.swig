{% extends '_layout.swig' %}
{% import '_macro/sidebar.swig' as sidebar_template with context %}

  {% block title %}
    {%- set page_title_suffix = ' | ' + title %}

    {%- if page.type === 'categories' and not page.title %}
      {{- __('title.category') + page_title_suffix }}
    {%- elif page.type === 'tags' and not page.title %}
      {{- __('title.tag') + page_title_suffix }}
    {%- elif page.type === 'schedule' and not page.title %}
      {{- __('title.schedule') + page_title_suffix }}
    {%- else %}
      {{- page.title + page_title_suffix }}
    {%- endif %}
  {% endblock %}

{% block content %}

  <div class="posts-expand">
    {##################}
    {### PAGE BLOCK ###}
    {##################}
    <div class="post-block" lang="{{ page.lang or page.language or config.language }}">
      {% include '_partials/page/page-header.swig' %}
      {#################}
      {### PAGE BODY ###}
      {#################}
      <div class="post-body{%- if page.direction and page.direction.toLowerCase() === 'rtl' %} rtl{%- endif %}">
        {%- if page.type === 'tags' %}
          <div class="tag-cloud">
            <div class="tag-cloud-title">
              {{ _p('counter.tag_cloud', site.tags.length) }}
            </div>
            {# tagcanvas plugin 球型云标签 #}
            {% include '_partials/tagcanvas.swig' %}
          </div>
        {% elif page.type === 'categories' %}
          <div class="category-all-page">
            <div class="category-all-title">
              {{ _p('counter.categories', site.categories.length) }}
            </div>
            <div class="category-all">
              {{ list_categories() }}
            </div>
          </div>
        {% elif page.type === 'schedule' %}
          <div class="event-list">
          </div>
          {% include '_scripts/pages/schedule.swig' %}
        {% else %}
          {{ page.content }}
        {%- endif %}
      </div>
      {#####################}
      {### END PAGE BODY ###}
      {#####################}
    </div>
    {% include '_partials/page/breadcrumb.swig' %}
    {######################}
    {### END PAGE BLOCK ###}
    {######################}
  </div>

{% endblock %}

{% block sidebar %}
  {{ sidebar_template.render(true) }}
{% endblock %}
