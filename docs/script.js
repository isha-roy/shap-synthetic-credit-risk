/* ============================================
   SCRIPT — Scroll Spy, Reveal Animations, TOC
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
  // --- Scroll Spy ---
  const tocLinks = document.querySelectorAll('.toc-nav a');
  const sections = document.querySelectorAll('.section[id]');

  const observerOptions = {
    root: null,
    rootMargin: '-20% 0px -70% 0px',
    threshold: 0
  };

  const scrollObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        tocLinks.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => scrollObserver.observe(section));

  // --- Section Reveal on Scroll ---
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll('.section').forEach(s => revealObserver.observe(s));

  // --- TOC Toggle (Desktop & Mobile) ---
  const tocToggle = document.querySelector('.toc-toggle');
  const tocSidebar = document.querySelector('.toc-sidebar');
  const bodyEl = document.body;

  if (tocToggle && tocSidebar) {
    // Set initial text content based on window size
    if (window.innerWidth > 768) {
      tocToggle.textContent = '✕';
    } else {
      tocToggle.textContent = '☰';
    }

    tocToggle.addEventListener('click', () => {
      if (window.innerWidth > 768) {
        bodyEl.classList.toggle('sidebar-collapsed');
        tocToggle.textContent = bodyEl.classList.contains('sidebar-collapsed') ? '☰' : '✕';
      } else {
        tocSidebar.classList.toggle('open');
        tocToggle.textContent = tocSidebar.classList.contains('open') ? '✕' : '☰';
      }
    });

    // Close TOC on link click (mobile)
    tocLinks.forEach(link => {
      link.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
          tocSidebar.classList.remove('open');
          tocToggle.textContent = '☰';
        }
      });
    });

    // Handle window resize resets
    window.addEventListener('resize', () => {
      if (window.innerWidth > 768) {
        tocSidebar.classList.remove('open');
        if (bodyEl.classList.contains('sidebar-collapsed')) {
          tocToggle.textContent = '☰';
        } else {
          tocToggle.textContent = '✕';
        }
      } else {
        if (tocSidebar.classList.contains('open')) {
          tocToggle.textContent = '✕';
        } else {
          tocToggle.textContent = '☰';
        }
      }
    });
  }

  // --- Back to Top ---
  const backToTop = document.querySelector('.back-to-top');
  if (backToTop) {
    window.addEventListener('scroll', () => {
      backToTop.classList.toggle('visible', window.scrollY > 500);
    }, { passive: true });

    backToTop.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // --- Smooth scroll for anchor links ---
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', e => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
});
