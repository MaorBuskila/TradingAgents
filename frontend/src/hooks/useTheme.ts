/**
 * useTheme — manages light / dark / system theme preference.
 *
 * Persists to localStorage as 'ta_theme'.
 * Applies by setting data-theme="light"|"dark" on <html>,
 * which wins over the OS @media prefers-color-scheme block.
 * "system" removes the attribute so the OS takes over.
 */
import { useState, useEffect } from 'react'

export type Theme = 'light' | 'dark' | 'system'

const STORAGE_KEY = 'ta_theme'

function applyTheme(theme: Theme) {
  const root = document.documentElement
  if (theme === 'system') {
    root.removeAttribute('data-theme')
  } else {
    root.setAttribute('data-theme', theme)
  }
}

function readStored(): Theme {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
  } catch { /* private browsing */ }
  return 'system'
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = readStored()
    // Apply immediately (before first render) to avoid flash
    applyTheme(stored)
    return stored
  })

  useEffect(() => {
    applyTheme(theme)
    try { localStorage.setItem(STORAGE_KEY, theme) } catch { /* ignore */ }
  }, [theme])

  return { theme, setTheme }
}
