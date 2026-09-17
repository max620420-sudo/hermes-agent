import { describe, expect, it } from 'vitest'

import {
  advanceVaultSaveLoginPrompt,
  claimVaultCodePrompt,
  getOverlayState,
  patchOverlayState,
  resetOverlayState,
  submitVaultSaveLoginPrompt
} from '../app/overlayStore.js'
import { vaultCodePromptKey, vaultSaveLoginPromptKey } from '../components/appOverlays.js'

describe('vault prompt instance keys', () => {
  it('changes when a different login-save request replaces the current request', () => {
    expect(vaultSaveLoginPromptKey('request-a', 'identifier')).not.toBe(
      vaultSaveLoginPromptKey('request-b', 'identifier')
    )
  })

  it('changes between identifier and password phases', () => {
    expect(vaultSaveLoginPromptKey('request-a', 'identifier')).not.toBe(
      vaultSaveLoginPromptKey('request-a', 'password')
    )
  })

  it('changes when a different verification-code request replaces the current request', () => {
    expect(vaultCodePromptKey('request-a')).not.toBe(vaultCodePromptKey('request-b'))
  })
})

describe('advanceVaultSaveLoginPrompt', () => {
  const initial = {
    identifier: '',
    origin: 'https://example.com',
    requestId: 'request-a',
    site: 'example.com',
    step: 'identifier' as const
  }

  const passwordStep = { ...initial, identifier: 'employee', step: 'password' as const }

  it('does not resurrect an expired request', () => {
    resetOverlayState()
    patchOverlayState({ vaultSaveLogin: null })

    advanceVaultSaveLoginPrompt(passwordStep)

    expect(getOverlayState().vaultSaveLogin).toBeNull()
  })

  it('does not replace a newer request', () => {
    resetOverlayState()
    patchOverlayState({ vaultSaveLogin: { ...initial, requestId: 'request-b' } })

    advanceVaultSaveLoginPrompt(passwordStep)

    expect(getOverlayState().vaultSaveLogin?.requestId).toBe('request-b')
  })

  it('advances the matching live request', () => {
    resetOverlayState()
    patchOverlayState({ vaultSaveLogin: initial })

    advanceVaultSaveLoginPrompt(passwordStep)

    expect(getOverlayState().vaultSaveLogin).toEqual(passwordStep)
  })
})

describe('stale vault submissions', () => {
  it('does not submit or clear a replacement login-save request', () => {
    resetOverlayState()
    patchOverlayState({
      vaultSaveLogin: {
        identifier: 'new-user',
        origin: 'https://new.example',
        requestId: 'request-b',
        site: 'new.example',
        step: 'password'
      }
    })

    expect(submitVaultSaveLoginPrompt('request-a', 'password', 'old-secret')).toBeNull()
    expect(getOverlayState().vaultSaveLogin?.requestId).toBe('request-b')
  })

  it('ignores an identifier callback after the same request advances to password', () => {
    resetOverlayState()
    patchOverlayState({
      vaultSaveLogin: {
        identifier: '',
        origin: 'https://example.com',
        requestId: 'request-a',
        site: 'example.com',
        step: 'identifier'
      }
    })

    expect(submitVaultSaveLoginPrompt('request-a', 'identifier', 'employee')?.kind).toBe('continue')
    expect(submitVaultSaveLoginPrompt('request-a', 'identifier', 'employee')).toBeNull()
    expect(getOverlayState().vaultSaveLogin).toMatchObject({
      identifier: 'employee',
      requestId: 'request-a',
      step: 'password'
    })
  })

  it('ignores a password callback after the same request resets to identifier', () => {
    resetOverlayState()
    patchOverlayState({
      vaultSaveLogin: {
        identifier: '',
        origin: 'https://example.com',
        requestId: 'request-a',
        site: 'example.com',
        step: 'identifier'
      }
    })

    expect(submitVaultSaveLoginPrompt('request-a', 'password', 'old-secret')).toBeNull()
    expect(getOverlayState().vaultSaveLogin).toMatchObject({
      identifier: '',
      requestId: 'request-a',
      step: 'identifier'
    })
  })

  it('does not claim or clear a replacement verification-code request', () => {
    resetOverlayState()
    patchOverlayState({ vaultCode: { requestId: 'code-b', site: 'new.example' } })

    expect(claimVaultCodePrompt('code-a')).toBe(false)
    expect(getOverlayState().vaultCode?.requestId).toBe('code-b')
  })

  it('claims and clears only the matching verification-code request', () => {
    resetOverlayState()
    patchOverlayState({ vaultCode: { requestId: 'code-a', site: 'example.com' } })

    expect(claimVaultCodePrompt('code-a')).toBe(true)
    expect(getOverlayState().vaultCode).toBeNull()
  })
})