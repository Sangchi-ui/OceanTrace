import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'OceanTrace Command Dashboard',
  description: 'Lagrangian transport hindcast and forecast engine',
}

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {children}
    </div>
  )
}
