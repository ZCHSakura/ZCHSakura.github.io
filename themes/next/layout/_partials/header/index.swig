{{ partial('_partials/header/brand.swig', {}, {cache: theme.cache.enable}) }}

<nav class="site-nav">
  {{ partial('_partials/header/menu.swig', {}, {cache: theme.cache.enable}) }}
</nav>

{{- next_inject('header') }}
