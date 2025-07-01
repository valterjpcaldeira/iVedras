const BASE_URL = 'http://web-production-830b1.up.railway.app'; // Change if backend runs elsewhere

export async function getComplaints() {
  const res = await fetch(`${BASE_URL}/complaints`);
  if (!res.ok) throw new Error('Erro ao obter queixas');
  return await res.json();
}

export async function submitComplaint(data) {
  const res = await fetch(`${BASE_URL}/complaints`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Erro ao submeter queixa');
  return await res.json();
}

export async function classifyComplaint(text) {
  const res = await fetch(`${BASE_URL}/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error('Erro ao classificar queixa');
  return await res.json();
} 